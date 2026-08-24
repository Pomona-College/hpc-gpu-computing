---
title: "Instructor Notes"
---

## Workshop Overview

This workshop introduces researchers to GPU computing on the Sagehen cluster at Pomona College. It covers GPU hardware, CUDA programming concepts, using GPU-accelerated Python frameworks (PyTorch, TensorFlow), and best practices for efficient GPU utilization.

**Total time:** Approximately 3.5--4 hours (including breaks)
**Target audience:** Researchers with some HPC experience who want to leverage GPU acceleration
**Prerequisites:** Workshops 0 (Intro to HPC) and 9 (SLURM Job Scheduling), or equivalent experience

## Pre-Workshop Preparation

### Technical Setup

1. **Verify GPU node availability:** Check that at least 1--2 GPUs are free during the workshop with `sinfo -p gpu --Node -o "%N %G %C %m %t"`
2. **Test module loads:** Confirm `module load cuda` and `module load miniconda3` work correctly
3. **Prepare a shared conda environment** with PyTorch and TensorFlow pre-installed (participants may not have time to install during the workshop)
4. **Pre-stage example datasets** in a shared `/bigdata/` location so participants do not need to download during class
5. **Test all code examples** on each GPU type (A100, L40S, RTX PRO 6000) to verify compatibility

### Logistics

- Book a computer lab with SSH/browser access, or ensure all participants have laptops
- Have a backup plan if GPU nodes are fully allocated (pre-recorded demos, screenshots)
- Prepare printed or digital copies of the reference card

## Episode-by-Episode Guide

### Episode 1: Introduction to GPU Computing (30 min teaching, 15 min exercises)

**Key concepts:** CPU vs. GPU architecture, parallelism, when GPUs help vs. hinder

**Teaching tips:**
- Start with a visual comparison of CPU (few powerful cores) vs. GPU (thousands of simple cores)
- Use the analogy: CPU is like a sports car (fast for one trip), GPU is like a bus fleet (moves many people at once)
- Show a live `nvidia-smi` demo on a GPU node to make it tangible
- Emphasize that GPUs are NOT always faster -- overhead matters for small problems

**Common questions:**
- "Can I use GPUs for my R code?" -- Generally no, unless using specific GPU-enabled packages like `gpuR`. Most R workloads benefit more from CPU parallelism
- "How do I know if my code will benefit from GPUs?" -- Look for operations on large matrices, deep learning training, or embarrassingly parallel computations

### Episode 2: GPU Hardware on Sagehen (20 min teaching, 10 min exercises)

**Key concepts:** A100/L40S/RTX PRO 6000 specs (10 GPUs total), VRAM, memory bandwidth, tensor cores

**Teaching tips:**
- Show the physical layout: which nodes have which GPUs
- Demonstrate `nvidia-smi` output and explain each field
- Discuss VRAM as the primary constraint for model size
- Explain why A100 80GB is preferred for large models

**Demo:** Run `nvidia-smi` on different GPU types and compare output

### Episode 3: CUDA and GPU Programming Basics (35 min teaching, 20 min exercises)

**Key concepts:** CUDA toolkit, kernels, threads, blocks, grids

**Teaching tips:**
- Focus on conceptual understanding, not deep CUDA C programming
- Most researchers will use PyTorch/TensorFlow rather than raw CUDA
- Show a simple CUDA program but emphasize that frameworks abstract this away
- The `nvcc --version` command is important for debugging version mismatches

### Episode 4: GPU Computing with Python (40 min teaching, 25 min exercises)

**Key concepts:** PyTorch `.cuda()`, TensorFlow GPU detection, data movement, mixed precision

**Teaching tips:**
- This is the most hands-on episode -- allocate extra time
- Have participants run interactive GPU sessions: `srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash`
- Walk through moving tensors to GPU and back
- Demonstrate the speed difference with a matrix multiplication benchmark
- Show how to check if code is actually using the GPU

**Common pitfall:** Participants forgetting to move BOTH the model AND data to GPU

### Episode 5: Multi-GPU and Distributed Training (35 min teaching, 15 min exercises)

**Key concepts:** DataParallel, DistributedDataParallel, multi-GPU SLURM scripts

**Teaching tips:**
- This is advanced -- adjust depth based on audience
- Focus on practical SLURM scripts for multi-GPU jobs
- Remind participants of the 4 GPU per account limit
- Show how to verify all GPUs are being used with `nvidia-smi`

### Episode 6: Performance Optimization (30 min teaching, 15 min exercises)

**Key concepts:** Mixed precision, memory management, profiling, data loading

**Teaching tips:**
- Mixed precision (FP16/BF16) is the single biggest easy optimization
- Show `torch.cuda.memory_summary()` for debugging memory issues
- Discuss data loading bottlenecks (CPU-bound preprocessing, I/O from `/rhome` vs. `/scratch`)
- Recommend `/scratch` or `/tmpfs` for training data I/O

### Episode 7: Best Practices and Troubleshooting (25 min teaching, 10 min exercises)

**Key concepts:** Resource etiquette, checkpointing, common errors

**Teaching tips:**
- Emphasize responsible GPU use: release resources when done, use `--time` limits
- Show how to set up model checkpointing to survive job timeouts
- Walk through common error messages and their solutions
- End with a Q&A session

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "No GPU nodes available" | Check `sinfo -p gpu`; consider off-peak hours or using L40S |
| "CUDA out of memory" | Reduce batch size, enable gradient checkpointing, use mixed precision |
| Slow training despite GPU | Check data loading (num_workers), ensure data is on `/scratch` |
| Module conflicts | Use `module purge` then load modules in correct order |
| Conda environment issues | Pre-build a shared environment; have a backup `.yml` file |

## Assessment Suggestions

- Have participants submit a GPU batch job and show the output
- Ask participants to compare CPU vs. GPU timing for a matrix multiplication
- Quiz on when to use GPUs vs. CPUs for different workload types

## Post-Workshop Resources

- Point participants to the learner reference card
- Encourage joining office hours for follow-up GPU questions
- Share links to PyTorch and TensorFlow GPU tutorials
- Contact: [its-hpc@pomona.edu](mailto:its-hpc@pomona.edu)
