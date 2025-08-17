import torch
x = torch.randint(0, 2, (16 * 1024,), dtype=torch.uint16, device="cuda")
out = torch.sort(x)
print(out)
