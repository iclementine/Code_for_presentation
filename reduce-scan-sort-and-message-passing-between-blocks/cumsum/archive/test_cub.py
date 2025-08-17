import torch
a = torch.randn((4566778,), device="cuda")
b = torch.cumsum(a, 0)
