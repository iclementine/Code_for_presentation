import torch

def generate_matrix(m: int, n: int , dtype: torch.dtype):
    if dtype.is_complex:
        raise TypeError
    elif dtype == torch.bool:
        x = torch.randint(0, 2, (m, n), dtype=dtype, device="cuda")
    elif dtype.is_floating_point:
        x = torch.randn((m, n), dtype=dtype, device="cuda")
    elif dtype.is_signed:
        iinfo = torch.iinfo(dtype)
        x = torch.randint(iinfo.min, iinfo.max, (m, n), dtype=dtype, device="cuda")
    else:
        iinfo = torch.iinfo(dtype)
        M = min(10000, iinfo.max)
        x = torch.randint(0, M, (m, n), dtype=dtype, device="cuda")
    return x
