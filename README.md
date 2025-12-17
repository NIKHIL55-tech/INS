# INS


import torch
import torch.nn as nn
import torch.profiler

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:

    for x, y in train_loader:   # runs over ALL batches
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10
))
