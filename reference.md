---
title: "Reference"
---

## GPU Hardware on Sagehen

### GPU Partition Overview

| GPU Model | Architecture | VRAM | FP32 (TFLOPS) | Tensor Cores | Quantity |
|-----------|-------------|------|---------------|--------------|----------|
| NVIDIA A100 | Ampere | 80 GB HBM2e | 19.5 | Yes (3rd gen) | 4 |
| NVIDIA L40S | Ada Lovelace | 48 GB GDDR6 | 91.6 | Yes (4th gen) | 4 |
| NVIDIA RTX PRO 6000 | Blackwell | 96 GB GDDR7 ECC | ~125 | Yes (5th gen) | 2 |

**Total: 10 GPUs** (confirmed by Andrew Wilson, May 2026).

### Account Limits

- **Maximum GPUs per account:** 4 simultaneous
- **Maximum CPU cores per account:** 512
- **Maximum submitted jobs per account:** 5,000

## SLURM GPU Commands

### Requesting GPUs

```bash
# Request any available GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Request a specific GPU type
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:l40s:1
#SBATCH --gres=gpu:l40s:1

# Request multiple GPUs
#SBATCH --gres=gpu:a100:2

# Interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

### Monitoring GPU Jobs

```bash
# Check GPU utilization on your allocated node
nvidia-smi

# Continuous monitoring (updates every 2 seconds)
nvidia-smi -l 2

# Compact monitoring output
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Check your running jobs
squeue -u $USER

# Check GPU partition availability
sinfo -p gpu --Node -o "%N %G %C %m %t"
```

## CUDA Module Commands

```bash
# List available CUDA versions
module spider cuda

# Load default CUDA
module load cuda

# Load specific CUDA version
module load cuda/12.2.1

# Check CUDA version
nvcc --version

# Check NVIDIA driver version
nvidia-smi | head -3
```

## Python GPU Frameworks

### PyTorch

```python
import torch

# Check CUDA availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Move tensor to GPU
x = torch.randn(1000, 1000).cuda()

# Specify device
device = torch.device('cuda:0')
model = MyModel().to(device)
```

### TensorFlow

```python
import tensorflow as tf

# Check GPU availability
print(tf.config.list_physical_devices('GPU'))

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### CUDA C/C++

```bash
# Compile CUDA code
nvcc -o my_program my_program.cu

# Compile with specific architecture
nvcc -arch=sm_80 -o my_program my_program.cu  # A100
nvcc -arch=sm_89 -o my_program my_program.cu  # L40S
nvcc -arch=sm_89 -o my_program my_program.cu  # L40S
nvcc -arch=sm_120 -o my_program my_program.cu  # RTX PRO 6000 (Blackwell)
```

## Common SLURM Batch Script Template

```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=gpu_job_%j.out
#SBATCH --error=gpu_job_%j.err

# Load modules
module load cuda
module load miniconda3

# Activate conda environment
conda activate my_gpu_env

# Run GPU code
python train_model.py
```

## Storage Recommendations for GPU Workloads

| Storage | Path | Use For | Speed |
|---------|------|---------|-------|
| Home | `/rhome/<myusername>` | Code, scripts, configs | Standard |
| Big Data | `/bigdata/lab/<labname>` | Datasets, model checkpoints | Standard |
| Scratch | `/scratch/$USER/$SLURM_JOB_ID` | Training data I/O, temp files | Fast (SSD) |
| Tmpfs | `/tmpfs/$USER/$SLURM_JOB_ID` | Fastest I/O, small datasets | Fastest (RAM) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-smi` not found | Load CUDA module: `module load cuda` |
| No GPU detected in PyTorch | Ensure `--gres=gpu:N` in SLURM script |
| CUDA out of memory | Reduce batch size, use mixed precision, or request a larger GPU |
| CUDA version mismatch | Match PyTorch/TF CUDA version with loaded module |
| Job pending for GPU | Check `sinfo -p gpu` for availability; consider using L40S |

## Contact

For GPU-related questions or issues, contact the HPC team at [its-hpc@pomona.edu](mailto:its-hpc@pomona.edu).

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
