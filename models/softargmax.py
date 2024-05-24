import numpy as np
import torch
import torch.nn as nn


def softargmax2d(input_tensor, scaling_factor_beta=100, dtype=torch.float32):
    *_, h, w = input_tensor.shape

    input_tensor = input_tensor.reshape(*_, h * w)
    input_tensor = nn.functional.softmax(scaling_factor_beta * input_tensor, dim=-1)

    # generate mesh coordinate grids with col_indices & row_indices. This grid contains normalized coordinates
    # between 0 and 1
    col_indices, row_indices = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    # convert to pytorch tensors
    row_indices = torch.tensor(np.reshape(row_indices, (-1, h * w)))
    col_indices = torch.tensor(np.reshape(col_indices, (-1, h * w)))

    device = input_tensor.get_device()
    if device >= 0:
        row_indices = row_indices.to(device)
        col_indices = col_indices.to(device)

    row_result_grid = torch.sum((h - 1) * input_tensor * row_indices, dim=-1)
    col_result_grid = torch.sum((w - 1) * input_tensor * col_indices, dim=-1)

    result_grid = torch.stack([row_result_grid, col_result_grid], dim=-1)

    return result_grid.type(dtype)


def softargmax1d(input_tensor, scaling_factor_beta=100, dtype=torch.float32):
    *_, n = input_tensor.shape
    input_tensor = nn.functional.softmax(scaling_factor_beta * input_tensor, dim=-1)
    indices = torch.linspace(0, 1, n)
    result_grid = torch.sum((n - 1) * input_tensor * indices, dim=-1)
    return result_grid.type(dtype)
