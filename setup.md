---
title: "Setup"
---

## Prerequisites

Before attending this workshop, you should:

1. **Have an active Sagehen HPC account.** If you do not yet have one, request access via the [HPC account request form](https://servicedesk.pomona.edu/support/catalog/items/83) (or by emailing [its-hpc@pomona.edu](mailto:its-hpc@pomona.edu)). Allow 1--2 business days for account provisioning.

2. **Complete the Introduction to HPC Systems workshop** (Workshop 0) or have equivalent experience with basic Linux commands, SSH access, and SLURM job submission on Sagehen.

3. **Complete the SLURM Job Scheduling workshop** (Workshop 9) or be comfortable writing and submitting batch scripts with `sbatch`.

4. **Be familiar with at least one programming language** commonly used in GPU computing (Python, C/C++, or CUDA).

## Accessing Sagehen

You can access the Sagehen cluster in two ways:

### Option 1: SSH (Terminal Access)

```bash
ssh <myusername>@sagehen.hpc.pomona.edu
```

You will be prompted for your Pomona College credentials and DUO Multi-Factor Authentication.

### Option 2: OnDemand Portal (Browser Access)

Navigate to [https://ondemand.hpc.pomona.edu/](https://ondemand.hpc.pomona.edu/) and log in with your Pomona credentials. OnDemand provides a web-based terminal, file manager, and JupyterLab for interactive GPU work.

## GPU Resources on Sagehen

Sagehen has **10 GPUs across the cluster** (confirmed May 2026; see README.md for hardware breakdown) in the `gpu` partition with the following accelerators:

| GPU Model | VRAM | Quantity | Best For |
|-----------|------|----------|----------|
| NVIDIA A100 | 80 GB | 4 | Large deep learning models, mixed-precision training |
| NVIDIA L40S | 48 GB | 4 | Inference, visualization, moderate training |
| NVIDIA RTX PRO 6000 | 96 GB | 2 | Large memory workloads, professional rendering, mid-large model training |

::::::::::::::::::::::::::::::::::::::: callout

## Account Limits

Each account/group is limited to a maximum of **4 GPUs** simultaneously. Plan your experiments accordingly and release GPU resources when you are done.

:::::::::::::::::::::::::::::::::::::::::::::::

## Loading GPU Software

Sagehen uses the **Lmod** module system to manage software. To set up a GPU computing environment:

```bash
# Check available CUDA versions
module spider cuda

# Load CUDA toolkit
module load cuda

# Load Python with GPU support (PyTorch, TensorFlow)
module load miniconda3

# Verify GPU access (on a GPU node only)
nvidia-smi
```

::::::::::::::::::::::::::::::::::::::: callout

## Important: GPUs Are Only on Compute Nodes

The head node (login node) does **not** have GPUs. You must request a GPU node via SLURM before running `nvidia-smi` or any GPU code. Use an interactive session for testing:

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

:::::::::::::::::::::::::::::::::::::::::::::::

## Pre-Workshop Checklist

Before the workshop begins, please verify the following:

- [ ] You can SSH into `sagehen.hpc.pomona.edu` or log into OnDemand
- [ ] You can load the CUDA module: `module load cuda`
- [ ] You can load miniconda3: `module load miniconda3`
- [ ] You understand how to submit a basic SLURM batch job

## Getting Help

If you encounter any issues during setup, contact the HPC team:

- **Email:** [its-hpc@pomona.edu](mailto:its-hpc@pomona.edu)
- **Response time:** Standard requests are addressed within 1--2 business days

<!-- highlight <labname>/<myusername> placeholders in code blocks; remove if the varnish theme handles this natively -->
<script>(function(){var CSS='.sh-placeholder{color:#c2410c;font-weight:700}[data-bs-theme="dark"] .sh-placeholder,html.dark .sh-placeholder{color:#fdba74}@media (prefers-color-scheme: dark){[data-bs-theme="auto"] .sh-placeholder{color:#fdba74}}';var RX=/<labname>|<myusername>/g;function firstMatch(el){var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null),nodes=[],full='';while(w.nextNode()){nodes.push({n:w.currentNode,s:full.length});full+=w.currentNode.nodeValue;}RX.lastIndex=0;var m;while((m=RX.exec(full))){var s=m.index,e=s+m[0].length,inSpan=false,parts=[];for(var j=0;j<nodes.length;j++){var ns=nodes[j].s,ne=ns+nodes[j].n.nodeValue.length;if(ne<=s||ns>=e)continue;parts.push({node:nodes[j].n,a:Math.max(s-ns,0),b:Math.min(e-ns,nodes[j].n.nodeValue.length)});var p=nodes[j].n.parentNode;while(p&&p!==el){if(p.classList&&p.classList.contains('sh-placeholder')){inSpan=true;break;}p=p.parentNode;}}if(!inSpan&&parts.length)return parts;}return null;}function wrapParts(parts){for(var i=parts.length-1;i>=0;i--){var t=parts[i].node,txt=t.nodeValue,a=parts[i].a,b=parts[i].b;var span=document.createElement('span');span.className='sh-placeholder';span.textContent=txt.slice(a,b);var f=document.createDocumentFragment();if(a>0)f.appendChild(document.createTextNode(txt.slice(0,a)));f.appendChild(span);if(b<txt.length)f.appendChild(document.createTextNode(txt.slice(b)));t.parentNode.replaceChild(f,t);}}function run(){var st=document.createElement('style');st.textContent=CSS;document.head.appendChild(st);document.querySelectorAll('pre,code').forEach(function(el){var guard=0,parts;while((parts=firstMatch(el))&&guard++<500){wrapParts(parts);}});}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',run);}else{run();}})();</script>
