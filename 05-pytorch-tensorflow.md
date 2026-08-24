---
title: "Training Models with PyTorch and TensorFlow"
teaching: 20
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you move models to GPU in PyTorch?
- How does TensorFlow handle GPU automatically?
- What are best practices for GPU training?
- How do you optimize GPU memory usage?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Train models efficiently on GPU
- Use distributed training for multi-GPU
- Monitor GPU usage during training
- Optimize memory and performance
::::::::::::::::::::::::::::::::::::::::::::::::

## PyTorch GPU Training

### Basic GPU Training

```python
import torch
import torch.nn as nn

# Move model to GPU
model = MyModel()
model = model.cuda()

# Create optimizer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for batch_data, batch_labels in train_loader:
        # Move data to GPU
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### Multi-GPU Training

```python
import torch.nn as nn
from torch.nn import DataParallel

# Distribute across multiple GPUs
model = DataParallel(model)
model = model.cuda()

# Rest of training loop unchanged
```

## TensorFlow GPU Training

### Automatic GPU Detection

```python
import tensorflow as tf

# TensorFlow automatically uses GPU
model = tf.keras.Sequential([...])

# Check GPU
print(tf.config.list_physical_devices('GPU'))

# Train (uses GPU automatically)
model.fit(train_data, train_labels, epochs=10)
```

### Manual GPU Placement

```python
with tf.device('/GPU:0'):
    model = tf.keras.Sequential([...])
    model.fit(train_data, train_labels)
```

## Monitoring GPU Usage

### During Training

In separate terminal:

```bash
watch -n 1 nvidia-smi
```

Or in Python:

```python
import subprocess

def print_gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', 
                            '--format=csv'], capture_output=True, text=True)
    print(result.stdout)
```

## Memory Optimization

### Batch Size Selection

```python
# Start small
batch_size = 16
# Check memory: nvidia-smi
# Increase if GPU has headroom
batch_size = 64
```

::::::::::::::::::::::::::::::::::::: callout

## Finding Optimal Batch Size

Larger batch sizes = better GPU utilization but more memory:
1. Start with batch_size = 16 or 32
2. Run `nvidia-smi` during training
3. If GPU memory > 80% used, batch size is good
4. If GPU memory < 50% used, increase batch size
5. If CUDA out of memory, reduce batch size

This simple loop finds the sweet spot for your hardware.

:::::::::::::::::::::::::::::::::::::::::::::::::

### Gradient Checkpointing (PyTorch)

Reduces memory for large models:

```python
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

### Mixed Precision Training

Faster, less memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, labels in train_loader:
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Fix the OOM Error

The following training script crashes with `RuntimeError: CUDA out of memory` on
an L40S (48GB) GPU. Identify **three** changes you would make to reduce GPU memory
usage, and show the modified code for each.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = LargeResNet().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset, batch_size=256, num_workers=1)

for epoch in range(100):
    for data, labels in train_loader:
        data = data.cuda()
        labels = labels.cuda()

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

::::::::::::::::::::::::::::::::::::: solution

## Solution

**Fix 1: Reduce batch size**

A batch size of 256 for a large model can exceed GPU memory. Cut it down:

```python
train_loader = DataLoader(dataset, batch_size=32, num_workers=1)
```

**Fix 2: Enable mixed precision training**

Use automatic mixed precision (AMP) to perform computations in float16 where
safe, roughly halving memory usage:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(100):
    for data, labels in train_loader:
        data = data.cuda()
        labels = labels.cuda()

        with autocast():
            outputs = model(data)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Fix 3: Use gradient checkpointing**

Trade compute time for memory by recomputing intermediate activations during the
backward pass instead of storing them all:

```python
from torch.utils.checkpoint import checkpoint_sequential

class LargeResNet(nn.Module):
    def forward(self, x):
        # Instead of: x = self.layers(x)
        x = checkpoint_sequential(self.layers, segments=4, input=x)
        return x
```

Each fix independently reduces GPU memory. Combining all three -- smaller batch
size, mixed precision, and gradient checkpointing -- can dramatically reduce
memory requirements, often enough to fit a model that was previously too large.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Move models to GPU: model.cuda()
- Move data to GPU: data.cuda()
- TensorFlow automatically uses GPU
- Monitor with nvidia-smi
- Optimize batch size for memory
- Use gradient checkpointing for large models
- Consider mixed precision for speed
::::::::::::::::::::::::::::::::::::::::::::::::
