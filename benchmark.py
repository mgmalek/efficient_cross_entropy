import argparse
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton
from tqdm.auto import tqdm

from modules import (
    FusedProjectionPlusCrossEntropyLoss,
    PyTorchProjectionPlusCrossEntropyLoss,
)


def _get_peak_memory_consumed(
    n_tokens, dim, n_classes, init_fn, dtype, device="cuda"
):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)

    feat = torch.randn(n_tokens, dim, dtype=dtype, requires_grad=True, device=device)
    targ = torch.randint(0, n_classes, (n_tokens,), device=device)

    forward_fn = init_fn()  # initialize module here so that we capture the memory consumption of the weights

    _ = forward_fn(feat, targ).mean().backward()

    torch.cuda.synchronize()
    final_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)

    peak_mem_consumed_bytes = final_peak_mem_bytes - initial_peak_mem_bytes
    peak_mem_consumed_gb = peak_mem_consumed_bytes / 1e9

    median_time_ms = triton.testing.do_bench(
        lambda: forward_fn(feat, targ).mean().backward(), return_mode="median"
    )

    return peak_mem_consumed_gb, median_time_ms


def run_test(
    line_configs, test_constant_kwargs, test_key, test_vals, mem_ax, time_ax, debug
) -> pd.DataFrame:
    if debug:
        test_vals = test_vals[:2]
        line_configs = line_configs[:2]

    constant_kwargs_label = ", ".join(f"{k}={v}" for k, v in test_constant_kwargs.items())

    mem_ax.set_title(f"Peak Memory Usage\n({constant_kwargs_label})")
    mem_ax.set_xlabel(test_key)

    time_ax.set_title(f"Median Wall Clock Time\n({constant_kwargs_label})")
    time_ax.set_xlabel(test_key)

    mem_ax.set_ylabel("Peak Memory Usage (GB)")
    time_ax.set_ylabel("Median Wall Clock Time (ms)")

    df_rows = []
    for line_config in tqdm(line_configs):
        line_label = ", ".join(f"{k}={v}" for k, v in line_config.items())

        line_config = copy(line_config)
        fn = line_config.pop("fn")

        peak_mems = []
        median_times = []

        for test_val in test_vals:
            params = {**test_constant_kwargs, test_key: test_val}
            dim = params["dim"]
            n_classes = params["n_classes"]
            n_tokens = params["n_tokens"]
            dtype = params["dtype"]

            if fn == "torch":
                module_cls = PyTorchProjectionPlusCrossEntropyLoss
            elif fn == "triton":
                module_cls = FusedProjectionPlusCrossEntropyLoss
            else:
                raise ValueError(f"Unknown {fn=}")

            init_fn = lambda: module_cls(dim, n_classes, **line_config).cuda().to(dtype)
            peak_mem, median_time = _get_peak_memory_consumed(
                n_tokens, dim, n_classes, init_fn=init_fn, dtype=dtype
            )
            peak_mems.append(peak_mem)
            median_times.append(median_time)
            df_rows.append(
                dict(
                    fn=fn,
                    dim=dim,
                    n_classes=n_classes,
                    n_tokens=n_tokens,
                    dtype=str(dtype),
                    peak_mem=peak_mem,
                    median_time=median_time,
                    n_loop_iters=line_config.get("n_loop_iters", None),
                )
            )

        mem_ax.plot(test_vals, peak_mems, "--o", label=line_label)
        time_ax.plot(test_vals, median_times, "--o", label=line_label)

    test_vals = [0.0] + test_vals

    for n in range(4):
        peak_mems_theory = []
        for test_val in test_vals:
            params = {**test_constant_kwargs, test_key: test_val}
            d = params["dim"]
            V = params["n_classes"]
            N = params["n_tokens"]
            dtype = params["dtype"]
            float_size =  {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}[dtype]

            peak_mems_theory.append(float_size * (
                2*N*d + 2*d*V + n*N*V # activations + weights + logits
            ) / 1e9)

        mem_ax.plot(test_vals, peak_mems_theory, "-", label=f'{float_size}*(2*N*D + 2*D*V + {n}*N*V)')

    for ax in (mem_ax, time_ax):
        ax.legend(loc="upper left")

    return pd.DataFrame(df_rows)


def benchmark(line_configs, output_dir, debug):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    _, (mem_axs, time_axs) = plt.subplots(2, 3, figsize=(20, 10), dpi=200)
    ax_mem_nclasses, ax_mem_tokens, ax_mem_dim = mem_axs
    ax_time_nclasses, ax_time_tokens, ax_time_dim = time_axs

    # Benchmark performance wrt n_classes
    test_key = "n_classes"
    test_vals = [2048, 4096, 8192, 16384, 32768, 65536]
    test_constant_kwargs = dict(dim=2048, n_tokens=8192, dtype=torch.float32)
    df_n_classes = run_test(
        line_configs,
        test_constant_kwargs,
        test_key,
        test_vals,
        ax_mem_nclasses,
        ax_time_nclasses,
        debug=debug,
    )
    df_n_classes.to_csv(output_dir / "n_classes.csv", index=False)

    # Benchmark performance wrt n_tokens
    test_key = "n_tokens"
    test_vals = [1024, 2048, 4096, 8192, 16384]
    test_constant_kwargs = dict(n_classes=32768, dim=2048, dtype=torch.float32)
    df_n_tokens = run_test(
        line_configs,
        test_constant_kwargs,
        test_key,
        test_vals,
        ax_mem_tokens,
        ax_time_tokens,
        debug=debug,
    )
    df_n_tokens.to_csv(output_dir / "n_tokens.csv", index=False)

    # Benchmark performance wrt dim
    test_key = "dim"
    test_vals = [1024, 2048, 4096, 8192, 16384, 32768]
    test_constant_kwargs = dict(n_classes=32768, n_tokens=8192, dtype=torch.float32)
    df_dim = run_test(
        line_configs,
        test_constant_kwargs,
        test_key,
        test_vals,
        ax_mem_dim,
        ax_time_dim,
        debug=debug,
    )
    df_dim.to_csv(output_dir / "dim.csv", index=False)

    plt.tight_layout()
    plt.savefig(output_dir / "plots.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Benchmark torch vs custom kernel
    line_configs = [
        dict(fn="torch"),
        dict(fn="triton", n_loop_iters=1),
        dict(fn="triton", n_loop_iters=2),
        dict(fn="triton", n_loop_iters=4),
        dict(fn="triton", n_loop_iters=8),
    ]
    benchmark(
        line_configs=line_configs,
        output_dir="./benchmark_data",
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
