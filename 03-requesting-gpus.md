---
title: "Requesting and Allocating GPUs"
teaching: 20
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions
- How do you request GPUs in SLURM jobs?
- What are GPU job parameters?
- How do you run interactive GPU sessions?
- How do you allocate multiple GPUs?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Submit GPU jobs with correct parameters
- Launch interactive GPU sessions
- Request multiple GPUs when needed
- Understand GPU partition and time limits
::::::::::::::::::::::::::::::::::::::::::::::::

## GPU Job Submission Basics

### Simple GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=gpu_%j.log

module load miniconda3
conda activate pytorch_env
python3 train_model.py
```

Key options:
- `--partition=gpu`: Must request GPU partition
- `--gres=gpu:1`: Number and type of GPUs
- `--time=01:00:00`: Max 720 hours per job

### Multi-GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module load miniconda3
conda activate pytorch_env
python3 -m torch.distributed.launch --nproc_per_node=2 train.py
```

### GPU Type Selection

```bash
# Any GPU
#SBATCH --gres=gpu:1

# Specific type
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:l40s:2
#SBATCH --gres=gpu:l40s:1
```

## Interactive GPU Sessions

### Launch Interactive GPU Shell

```bash
srun -p gpu --gres=gpu:1 --time=01:00:00 --pty bash -l
```

Now in interactive shell with GPU access:

```bash
nvidia-smi  # Check GPU
python3     # Start Python with GPU
```

### Jupyter on GPU

Through OnDemand:
1. Click Interactive Apps → Jupyter Notebook
2. Select GPU in launcher
3. Click Launch
4. Jupyter starts with GPU access

In job script:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

## Memory and Core Allocation

### CPU Allocation

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
```

For each GPU, allocate 4-8 CPU cores for:
- Data loading
- Preprocessing
- I/O operations

::::::::::::::::::::::::::::::::::::: callout

## Why CPU Cores Matter for GPU Jobs

Under-allocating CPUs creates a bottleneck:
- GPU sits idle waiting for data
- Data loading becomes the limiting factor
- Job runs slow despite having GPU

**Rule of thumb**: 4-8 CPU cores per GPU for efficient data feeding.
Monitor GPU utilization with `nvidia-smi` to verify CPU isn't bottlenecking.

:::::::::::::::::::::::::::::::::::::::::::::::::

### Memory Allocation

```bash
#SBATCH --mem=32G  # Total memory needed
```

Or per CPU:

```bash
#SBATCH --mem-per-cpu=8G
```

### Recommendation

Per GPU, request:
- 4-8 CPU cores
- 8-16 GB memory
- 1-4 hour time limit for testing

## Time Limits

GPU partition allows:
- Max 720 hours (30 days) per job
- Typical: 1-8 hours for training
- Recommended: Start with 1 hour for testing

```bash
# Test run
#SBATCH --time=00:30:00

# Standard training
#SBATCH --time=04:00:00

# Long training
#SBATCH --time=24:00:00
```

## Batch Scripts with Setup

Complete example:

```bash
#!/bin/bash
#SBATCH --job-name=deep_learning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

# Setup environment
module purge
module load miniconda3 cuda/12.2.1
conda activate pytorch_env

cd /bigdata/lab/<labname>/project

# Check GPU
echo "GPU allocation:"
nvidia-smi

# Run training
python3 train.py --gpu 0 --batch-size 32 --epochs 100

echo "Training complete"
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Write a GPU Job Script

Write a complete SLURM batch script that meets the following requirements:

- **Job name:** `pytorch_training`
- **GPUs:** 2 NVIDIA A100s
- **Time limit:** 4 hours
- **Memory:** 64 GB
- **CPU cores:** 8 per task (for data loading)
- **Software:** Load the `cuda/12.2.1` module and activate your `pytorch_env` conda environment
- **Command:** Run `python3 train.py --distributed`
- **Output:** Log file named with the job name and job ID

::::::::::::::::::::::::::::::::::::: solution

## Solution

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

# Load required modules
module purge
module load miniconda3 cuda/12.2.1
conda activate pytorch_env

# Verify GPU allocation
echo "Allocated GPUs:"
nvidia-smi

# Run distributed training
python3 train.py --distributed

echo "Training complete"
```

Key points:
- `--partition=gpu` is required for GPU access on Sagehen.
- `--gres=gpu:a100:2` requests exactly 2 A100 GPUs.
- `--cpus-per-task=8` provides enough CPU cores for parallel data loading.
- `--mem=64G` allocates 64 GB of system RAM (separate from GPU memory).
- `%x` expands to the job name and `%j` to the job ID in output filenames.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- GPU jobs require: --partition=gpu and --gres=gpu:N
- Request matching CPUs and memory for your GPU
- Max 720 hours per job; typical 1-8 hours
- Interactive sessions available for testing
- Multiple GPUs require distributed training setup
::::::::::::::::::::::::::::::::::::::::::::::::

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
