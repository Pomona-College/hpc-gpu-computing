---
title: "Introduction to GPU Computing"
teaching: 15
exercises: 5
---

:::::::::::::::::::::::::::::::::::::: questions
- What are GPUs and why use them?
- When is GPU acceleration beneficial?
- What workloads suit GPU computing?
- How do GPUs differ from CPUs?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Understand GPU architecture and benefits
- Identify GPU-suitable workloads
- Know when to use GPUs vs CPUs
- Understand cost-benefit tradeoffs
::::::::::::::::::::::::::::::::::::::::::::::::

## What is a GPU?

GPUs (Graphics Processing Units) are specialized processors optimized for:
- Parallel computation (thousands of threads simultaneously)
- High memory bandwidth (fast data access)
- Matrix operations and tensor computations
- Deep learning and machine learning

### CPU vs GPU

**CPUs**: Sequential computation, complex logic, few cores (4-128)
**GPUs**: Parallel computation, high throughput, thousands of cores

**Analogy**: 
- CPU is a Ferrari (fast, powerful, limited seats)
- GPU is a bus (slower per person, carries many passengers)

### Performance Gains

GPUs can provide 10x-100x speedup for suitable tasks:
- Deep learning training: 20-50x speedup
- Scientific simulations: 10-30x speedup
- Image processing: 5-20x speedup
- Data sorting: 5-10x speedup

## Sagehen GPU Hardware

Sagehen has **10 GPUs across multiple nodes** (confirmed May 2026): 4× A100 (80 GB), 4× L40S (48 GB), 2× RTX PRO 6000 Blackwell (96 GB). For full hardware breakdown see README.md or Workshop 16 episode 02.

### Node Configuration

- **NVIDIA A100** (×4): 80 GB HBM2e memory (highest performance)
- **NVIDIA L40S** (×4): 48 GB GDDR6 memory (deep learning optimized)
- **NVIDIA RTX PRO 6000 Blackwell** (×2): 96 GB GDDR7 ECC memory (the largest GPU memory on Sagehen)

10 GPUs available across the cluster

### GPU Specifications

| GPU | Memory | Performance | Quantity | Best For |
|-----|--------|-------------|----------|----------|
| A100 | 80 GB | Highest | 4 | Large models |
| L40S | 48 GB | Very High | 4 | Deep learning |
| RTX PRO 6000 | 96 GB ECC | Very High | 2 | Largest-memory models / ECC workloads |

### Choosing the Right GPU

- **Training the largest models (>80 GB)**: RTX PRO 6000 (96 GB) — the largest memory on Sagehen
- **Large models (48–80 GB)**: A100 80 GB or RTX PRO 6000
- **Deep learning research (up to 48 GB)**: L40S 48 GB
- **Standard ML**: L40S 48 GB
- **ECC memory needed**: RTX PRO 6000
- **Cost-conscious / prototyping**: L40S — the least contended of the three, so it is the sensible default for iterating

## GPU-Suitable Workloads

### Ideal for GPU

- Deep learning (training neural networks)
- Scientific simulations
- Image and video processing
- Matrix operations
- Data analytics at scale
- Physics simulations
- Climate modeling

### NOT Suitable for GPU

- Sequential code with little parallelism
- Complex control flow
- Random memory access patterns
- Very small data (transfer overhead > computation gain)
- Single-threaded algorithms

### Example: When GPU Helps

**Data science workflow**:
1. Load data: CPU (fast)
2. Preprocess: CPU (sequential steps)
3. Train model: GPU (thousands of iterations, matrix ops)
4. Evaluate: CPU (fast, little computation)

Only step 3 benefits from GPU; transfer overhead may dominate.

## GPU Programming Models

### CUDA (NVIDIA)
- Native NVIDIA programming
- Highest performance
- Requires C/C++ knowledge
- Not for beginners

### High-Level Libraries

Most users should use:
- **PyTorch**: Deep learning, Python, easy GPU support
- **TensorFlow**: Deep learning, industry standard
- **CuPy**: NumPy-like array computing on GPU
- **scikit-cuda**: scikit-learn acceleration

### Framework Automatic GPU Use

Modern frameworks auto-detect GPUs:

```python
import torch
# PyTorch automatically uses GPU if available
model = MyModel().cuda()  # Explicitly move to GPU
```

```python
import tensorflow as tf
# TensorFlow automatically uses GPU
gpus = tf.config.list_physical_devices('GPU')
```

## Getting Started with GPU on Sagehen

### Three Steps

1. Request GPU in job submission: `--gres=gpu:1`
2. Activate your GPU-enabled conda environment: `module load miniconda3 && conda activate pytorch_env` (no pytorch module exists)
3. Use framework normally; GPU automatically used

More details in subsequent episodes.

::::::::::::::::::::::::::::::::::::: callout

## Before Requesting GPU Time

GPU time is expensive and limited. Always:
1. **Test on CPU first** with small data
2. **Verify your code works** before GPU submission
3. **Estimate GPU benefits** for your workload
4. **Check if GPU will help** (not all code benefits equally)

This saves GPU allocation for researchers who truly need it.

:::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: GPU or CPU?

For each of the following computational tasks, decide whether it is better suited
for **GPU** or **CPU** execution. Briefly explain your reasoning.

1. Multiplying two 10,000 x 10,000 matrices
2. Parsing a 500 MB CSV file line by line
3. Training a convolutional neural network on 1 million images
4. Sorting a list of 100 integers
5. Applying a 3x3 convolution filter across a 4K image
6. Performing a recursive depth-first search on a tree with complex branching logic

::::::::::::::::::::::::::::::::::::: solution

## Solution

1. **Matrix multiplication -- GPU.** Matrix multiplication is a massively parallel
   operation (each output element is an independent dot product). GPUs with thousands
   of cores handle this 10-100x faster than CPUs.

2. **CSV parsing -- CPU.** Line-by-line file parsing is inherently sequential with
   complex string logic and branching. GPUs provide no benefit here.

3. **CNN training -- GPU.** Neural network training involves repeated matrix
   multiplications, convolutions, and gradient computations across large batches --
   exactly what GPUs excel at (20-50x speedup).

4. **Sorting 100 integers -- CPU.** The dataset is tiny. The overhead of transferring
   data to the GPU would exceed any computational benefit. A CPU sorts this in
   microseconds.

5. **Image convolution -- GPU.** Convolution applies the same operation independently
   to every pixel neighborhood -- a textbook data-parallel workload ideal for GPU.

6. **Recursive tree search -- CPU.** Recursive algorithms with irregular branching and
   random memory access patterns map poorly to GPU architecture. CPUs handle complex
   control flow far more efficiently.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- GPUs excel at parallel computation for thousands of operations
- Sagehen has 10 GPUs across three types: A100, L40S, and RTX PRO 6000
- Deep learning, simulations, and data processing benefit most
- Framework libraries (PyTorch, TensorFlow) handle GPU automatically
- Not all workloads benefit; profile before investing GPU time
::::::::::::::::::::::::::::::::::::::::::::::::
