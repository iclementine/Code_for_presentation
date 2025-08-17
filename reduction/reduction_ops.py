import triton
from triton import language as tl
import math

def normalize_axis(axis, ndim):
    assert (-ndim <= axis) and (axis < ndim), f"axis {axis} out of range for tensor with ndim={ndim}"
    return axis if axis > 0 else axis + ndim

def reverse_perm(perm):
    """Generate the reverse permutation given a permutation"""
    ndim = len(perm)
    rev = [0 for _ in range(ndim)]
    for s, t in enumerate(perm):
        rev[t] = s
    return rev

def move_reduction_axes_last(inp, dims):
    if isinstance(dims, int):
        dims = [dims]

    ndims = inp.ndim
    N = inp.numel()
    reduce_dims = [normalize_axis(i, ndims) for i in dims]
    num_red_dims = len(dims)
    num_batch_dims = ndims - num_red_dims

    perm = list(range(ndims))
    is_reduce = [0 for i in range(ndims)]
    for i in reduce_dims:
        is_reduce[i] = 1
    perm.sort(key=lambda i: is_reduce[i])

    strides = inp.stride()
    perm[: num_batch_dims] = sorted(perm[: num_batch_dims], key=lambda i: strides[i], reverse=True)
    perm[num_batch_dims:] = sorted(perm[num_batch_dims:], key=lambda i: strides[i], reverse=True)
    rev_perm = reverse_perm(perm)

    original_shape = inp.shape
    permuted_shape = [original_shape[i] for i in perm]
    return permuted_shape, perm, rev_perm


import torch
x = torch.randn(3, 4, 5, 6, 7).permute(0, 3, 4, 1, 2)
y = move_reduction_axes_last(x, 1)
