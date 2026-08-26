# GPU Computing on Sagehen HPC

Pomona College HPC Workshop Series

## Overview

This workshop introduces GPU computing on the Sagehen HPC cluster, covering GPU architecture, hardware capabilities, and practical training with PyTorch and TensorFlow. Participants learn when GPUs provide performance benefits, how to request GPU resources via SLURM, and how to optimize deep learning workloads. The workshop includes hands-on examples demonstrating 10-50x speedups for suitable computational tasks.

## Episodes

1. **Introduction to GPU Computing**: Understand GPU architecture, GPU vs CPU differences, performance gains for different workload types, and when to use GPUs.
2. **Sagehen GPU Hardware**: Learn about Sagehen's GPU inventory. Sagehen has 10 GPUs across multiple nodes: 4× NVIDIA A100 80 GB, 4× NVIDIA L40S 48 GB, 2× NVIDIA RTX PRO 6000 Blackwell 96 GB (confirmed by Andrew Wilson, May 2026).
3. **Requesting GPUs Through SLURM**: Submit GPU resource requests in batch scripts, specify GPU types and quantities, check GPU availability, and monitor GPU job queues.
4. **GPU Software Setup**: Configure CUDA environments, load GPU libraries and frameworks, verify GPU availability, and set up development environments for GPU computing.
5. **Training Models with PyTorch and TensorFlow**: Move models to GPU, use distributed training across multiple GPUs, implement efficient training loops, and optimize memory usage.
6. **Monitoring and Optimization**: Monitor GPU utilization and memory, identify performance bottlenecks, optimize GPU code, and troubleshoot common GPU issues.
7. **GPU Etiquette and Best Practices**: Share GPU resources fairly, avoid hogging GPUs, use batch queuing efficiently, and follow cluster guidelines for GPU usage.

## Prerequisites

- Active Sagehen HPC cluster account
- Familiarity with Linux command line and SSH
- Experience with SLURM job submission
- Basic Python programming knowledge
- Familiarity with machine learning frameworks (PyTorch or TensorFlow) is helpful

## Learning Objectives

After completing this workshop, learners will be able to:
- Understand GPU architecture and when GPU acceleration is beneficial
- Identify GPU-suitable workloads and estimate performance improvements
- Request GPU resources using SLURM job scripts
- Set up GPU computing environments with CUDA and ML frameworks
- Train and run models efficiently on Sagehen GPUs
- Monitor GPU performance and optimize memory usage
- Follow best practices for shared GPU resource usage

## Target Audience

Researchers using deep learning, machine learning, scientific simulations, image processing, or other computationally intensive workflows that benefit from GPU acceleration. This includes graduate students, postdocs, and faculty in STEM disciplines.

## Duration

Approximately 3-4 hours, including hands-on GPU training examples and performance demonstrations.

## Technical Requirements

- Sagehen HPC cluster account
- SSH access to Sagehen login nodes
- SLURM knowledge for job submission
- GPU-capable software: CUDA toolkit, PyTorch, TensorFlow, or cuPy
- GPU allocation in Sagehen resource scheduler
- Text editor for job scripts

## Contact

- **Email**: its-hpc@pomona.edu
- **Workshop Author**: Andrew Wilson, Director of Research Computing

## License

This workshop is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Wilson, A. (2026). *GPU Computing on Sagehen*. Pomona College ITS Research Computing.

## Acknowledgments

**Andrew Wilson** — Director of Research Computing and Digital Scholarship,
Pomona College. Workshop design and development.

**Andrei Motchenko** — testing, editing, cleanup and screenshots across the
Pomona College HPC Workshop Series.
