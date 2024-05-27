from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from modules import (
    FusedProjectionPlusCrossEntropyLoss,
    PyTorchProjectionPlusCrossEntropyLoss,
)


def fwd_bwd(module: nn.Module, x0: torch.Tensor, targ: torch.Tensor):
    x = x0.clone()
    x.requires_grad_(True)
    loss = 2.0 * module(x, targ)
    print(loss.shape)
    loss.mean().backward()
    grad = x.grad
    assert x.equal(x0)
    if isinstance(module, FusedProjectionPlusCrossEntropyLoss):
        proj_weight_grad = module.proj_weight.grad
    else:
        proj_weight_grad = module.proj.weight.grad
    return loss, grad, proj_weight_grad


@pytest.mark.parametrize("n_tokens", [8, 1536])
@pytest.mark.parametrize("n_classes", [8, 2048])
@pytest.mark.parametrize("dim", [8, 2048])
@pytest.mark.parametrize("n_loop_iters", [1, 2, 4])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
# @pytest.mark.parametrize("use_ignore_index", [True, False])
@pytest.mark.parametrize("use_ignore_index", [False])
@pytest.mark.parametrize("autocast", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("stddev", [1.0, 10.0])
def test_correctness(
    n_tokens, n_classes, dim, n_loop_iters, reduction, use_ignore_index, autocast, dtype, device, stddev
):
    torch.manual_seed(0)

    x = stddev * torch.randn(n_tokens, dim, device=device, dtype=dtype)
    targ = torch.randint(low=0, high=n_classes, size=(n_tokens,), device=device)

    ignore_index = -1
    if use_ignore_index:
        targ = torch.where(
            torch.rand(targ.shape, device=targ.device) < 0.5, targ, ignore_index
        )

    torch_module = PyTorchProjectionPlusCrossEntropyLoss(
        dim, n_classes, ignore_index=ignore_index, reduction=reduction,
    ).to(device, dtype=dtype)

    triton_module = FusedProjectionPlusCrossEntropyLoss(
        dim, n_classes, n_loop_iters, ignore_index=ignore_index, reduction=reduction
    ).to(device, dtype=dtype)

    torch_fp32_module = deepcopy(torch_module)

    assert triton_module.proj_weight.data.shape == torch_module.proj.weight.data.shape
    triton_module.proj_weight.data = torch_module.proj.weight.data

    with torch.cuda.amp.autocast(enabled=autocast, dtype=torch.bfloat16):
        torch_loss, torch_grad, torch_proj_weight_grad = fwd_bwd(torch_module, x, targ)
        triton_loss, triton_grad, triton_proj_weight_grad = fwd_bwd(triton_module, x, targ)
    
    assert torch_grad is not None
    assert torch_loss.dtype == triton_loss.dtype, (torch_loss.dtype, triton_loss.dtype)
    assert torch_grad.dtype == triton_grad.dtype, (torch_grad.dtype, triton_grad.dtype)
    assert torch_proj_weight_grad.dtype == triton_proj_weight_grad.dtype, (torch_proj_weight_grad.dtype, triton_proj_weight_grad.dtype)

    if autocast:
        # autocast correctness is validated by checking that the norm of the loss and gradients
        # between pytorch fp32 and pytorch autocast is similar to the norm of the loss and gradients
        # between pytorch fp32 and triton autocast
        torch_fp32_loss, torch_fp32_grad, torch_fp32_proj_weight_grad = fwd_bwd(torch_fp32_module, x, targ)

        torch_loss_norm = torch.linalg.norm(torch_fp32_loss - torch_loss).item()
        torch_grad_norm = torch.linalg.norm(torch_fp32_grad - torch_grad).item()
        torch_proj_weight_grad_norm = torch.linalg.norm(torch_fp32_proj_weight_grad - torch_proj_weight_grad).item()

        triton_loss_norm = torch.linalg.norm(torch_fp32_loss - triton_loss).item()
        triton_grad_norm = torch.linalg.norm(torch_fp32_grad - triton_grad).item()
        triton_proj_weight_grad_norm = torch.linalg.norm(torch_fp32_proj_weight_grad - triton_proj_weight_grad).item()

        assert triton_loss_norm < 2 * torch_loss_norm, (triton_loss_norm, torch_loss_norm)
        assert triton_grad_norm < 2 * torch_grad_norm, (triton_loss_norm, torch_loss_norm)
        assert triton_proj_weight_grad_norm < 2 * torch_proj_weight_grad_norm, (triton_loss_norm, torch_loss_norm)

    else:
        assert torch.allclose(torch_loss, triton_loss, rtol=1e-4)
        assert torch.allclose(torch_grad, triton_grad, atol=1e-3, rtol=1e-4)
        assert torch.allclose(torch_proj_weight_grad, triton_proj_weight_grad, atol=1e-2, rtol=1e-2)
