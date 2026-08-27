---
title: "Learner Profiles"
---

## Profile 1: Maria -- Neuroscience PhD Student

**Background:** Maria is a third-year PhD student in neuroscience. She has been training convolutional neural networks (CNNs) on brain imaging data using her laptop, but training takes days and she frequently runs out of memory with larger models.

**Experience:** Comfortable with Python and PyTorch. Has used Sagehen HPC for CPU-based data preprocessing (Workshops 0 and 9 completed). Has never used GPUs on a cluster before.

**Motivation:** She needs to train deeper models on larger datasets for her dissertation. Her advisor suggested using the A100 GPUs on Sagehen to speed up training from days to hours.

**Goals for this workshop:**
- Learn how to request GPU resources through SLURM
- Understand how to move her existing PyTorch training scripts to GPU
- Learn to use mixed precision training to fit larger models in memory
- Understand multi-GPU training for her largest experiments

## Profile 2: Dr. James Chen -- Chemistry Faculty

**Background:** Dr. Chen is a computational chemistry professor who uses molecular dynamics (MD) simulations. His lab has been running GROMACS and LAMMPS on CPU nodes, but he has heard that GPU-accelerated MD can be 10--50x faster.

**Experience:** Strong Linux and HPC background. Familiar with SLURM and module system. Has compiled scientific software from source before. Limited Python experience.

**Motivation:** He wants to run longer and larger molecular dynamics simulations for a recently funded NSF project. GPU acceleration would let his students explore more parameter space.

**Goals for this workshop:**
- Understand which GPU types are best for MD simulations
- Learn to compile GPU-enabled versions of GROMACS/LAMMPS
- Set up efficient batch scripts for GPU MD jobs
- Understand performance trade-offs between GPU types

## Profile 3: Alex Rivera -- Senior Undergraduate in Computer Science

**Background:** Alex is a senior CS major working on a capstone project involving natural language processing (NLP). They have taken a machine learning course that covered neural networks conceptually but have limited hands-on experience with GPU training.

**Experience:** Good programming skills in Python. Has used Sagehen for basic assignments. Familiar with the command line but new to SLURM and GPU computing.

**Motivation:** Their capstone project requires fine-tuning a pre-trained language model, which is impractical on a laptop. Their faculty advisor recommended using the HPC GPUs.

**Goals for this workshop:**
- Learn the basics of GPU computing concepts
- Understand how to set up a Python environment with GPU support on Sagehen
- Run their first GPU training job on the cluster
- Learn enough to fine-tune a language model for their capstone
