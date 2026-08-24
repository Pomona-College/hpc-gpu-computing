---
title: "Monitoring and Optimization"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you monitor GPU performance?
- What metrics matter for GPU workloads?
- How do you profile GPU code?
- What optimizations improve performance?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Monitor GPU utilization and memory
- Identify performance bottlenecks
- Profile GPU code efficiently
- Implement GPU optimizations
::::::::::::::::::::::::::::::::::::::::::::::::

## Monitoring GPU Resources

### nvidia-smi Command

```bash
# One-time check
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi --id=0

# Memory and utilization format
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
```

### Key Metrics

- **GPU Utilization**: Percentage of GPU computing (target: 80-100%)
- **Memory Used**: Current GPU memory usage
- **Memory Total**: Maximum available GPU memory
- **Temperature**: GPU heat (should be <80°C)
- **Power Usage**: Watts consumed

### In Python

```python
import torch

def print_gpu_usage():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")

print_gpu_usage()
```

## Performance Profiling

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Your training code
    model(input_data)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### TensorFlow Profiler

```python
import tensorflow as tf

# Profile a training step
tf.profiler.experimental.start('/tmp/logdir')
# Training code
tf.profiler.experimental.stop()
```

## Optimization Techniques

### 1. Increase Batch Size

```python
# Memory limited? Check GPU with nvidia-smi
batch_size = 128  # Larger batch = better GPU utilization
dataloader = DataLoader(dataset, batch_size=batch_size)
```

### 2. Data Loading Optimization

```python
# Efficient loading
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,          # Parallel loading
    pin_memory=True,        # Fast CPU→GPU transfer
    prefetch_factor=2       # Prefetch next batch
)
```

### 3. Mixed Precision

```python
from torch.cuda.amp import autocast
# Reduces memory, faster computation
with autocast():
    output = model(input)
```

### 4. Gradient Accumulation

```python
# Train with large effective batch without huge memory
accumulation_steps = 4
for i, (data, labels) in enumerate(dataloader):
    outputs = model(data)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Common Performance Issues

### Low GPU Utilization

Causes:
- Data loading is slow (CPU bottleneck)
- Small batch size
- I/O operations block computation

Solutions:
- Increase num_workers in DataLoader
- Larger batch size
- Pin memory, prefetch data

::::::::::::::::::::::::::::::::::::: callout

## GPU Utilization is the #1 Performance Indicator

If GPU utilization is below 50%, you're wasting GPU resources:

**Check with**: `nvidia-smi` during training

**If low utilization**:
1. Look at CPU cores: are they maxed out?
2. Check num_workers in DataLoader
3. Profile with PyTorch profiler to find bottleneck
4. Increase batch size to feed GPU faster

High GPU utilization = money well spent on HPC allocation.

:::::::::::::::::::::::::::::::::::::::::::::::::

### Out of Memory

Solutions:
- Reduce batch size
- Clear cache: torch.cuda.empty_cache()
- Use gradient checkpointing
- Use mixed precision
- Check for memory leaks

### Slow Training

Check:
- Is GPU being used? (nvidia-smi)
- What's the GPU utilization percentage?
- Is CPU saturating dataloader?
- Are there synchronization points?

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Diagnose Low Utilization

You run `nvidia-smi` during a PyTorch training job and see the following output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0    62W / 400W |  38420MiB / 81920MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
```

The GPU is only at **5% utilization** despite the model using 38 GB of memory.

1. What is the most likely cause of the low GPU utilization?
2. What specific code change would you make to fix it?

::::::::::::::::::::::::::::::::::::: solution

## Solution

**1. Most likely cause: data loading bottleneck**

The GPU has plenty of memory allocated (38 GB) but is sitting idle 95% of the time.
This almost always means the GPU finishes each batch computation quickly, then waits
for the CPU to load and preprocess the next batch. The data pipeline cannot keep up
with the GPU.

**2. Fix: increase DataLoader workers and enable pinned memory**

The default DataLoader uses a single worker process (`num_workers=0` or `1`), which
creates a serial bottleneck. Increase parallel workers and enable `pin_memory` for
faster CPU-to-GPU transfers:

```python
# Before (slow):
train_loader = DataLoader(dataset, batch_size=64, num_workers=1)

# After (fast):
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,        # Parallel data loading processes
    pin_memory=True,      # Faster CPU-to-GPU memory transfer
    prefetch_factor=2     # Prefetch next batches while GPU computes
)
```

With these changes, multiple CPU workers load and preprocess batches in parallel
while the GPU is computing, keeping the GPU fed with data. You should see GPU
utilization rise to 80-100%.

A good rule of thumb is 4-8 CPU workers per GPU. Make sure your SLURM job
requests enough CPU cores to match: `--cpus-per-task=8`.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Use nvidia-smi to monitor GPU utilization and memory
- Aim for 80-100% GPU utilization
- Profile code to identify bottlenecks
- Batch size, data loading, and precision affect performance
- Monitor memory to avoid OOM errors
::::::::::::::::::::::::::::::::::::::::::::::::
