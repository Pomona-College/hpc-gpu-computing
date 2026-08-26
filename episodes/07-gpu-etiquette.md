---
title: "GPU Etiquette and Best Practices"
teaching: 15
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions
- Why is GPU etiquette important?
- How long can you run GPU jobs?
- What are fair usage practices?
- How do you share GPU resources?
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Understand GPU resource sharing
- Practice fair usage of limited resources
- Know time and resource limits
- Implement GPU best practices
::::::::::::::::::::::::::::::::::::::::::::::::

## GPU Resource Limits

### Per-User Limits

- **Max concurrent GPUs**: Per-account limits apply
- **Max per job**: Check current policy
- **Max time**: 720 hours (30 days)
- **Queue priority**: Fair-share based on usage

### Cluster Limits

- **Total GPUs**: Limited shared resource
- **Peak utilization**: Usually >80%
- **High demand**: Especially A100 and L40S

## Fair Usage Practices

### 1. Request Appropriate Resources

```bash
# Not: 4x A100 80GB for testing
# Use L40S for testing

#SBATCH --gres=gpu:l40s:1  # For quick test
#SBATCH --time=00:30:00
```

### 2. Don't Hog Resources

```bash
# Bad: Run job for 30 days when only 4 hours needed
#SBATCH --time=720:00:00

# Good: Request actual time needed
#SBATCH --time=04:00:00
```

### 3. Avoid Interactive Idle Sessions

```bash
# Don't do this:
srun -p gpu --gres=gpu:a100:1 --time=24:00:00 --pty bash
# ... leave running overnight

# Instead: Close when not using
exit  # Release GPU immediately
```

::::::::::::::::::::::::::::::::::::: callout

## GPU Idle Time Costs Everyone

A single A100 left idle:
- Costs ~$2-3 per hour in compute time
- Blocks other researchers from using it
- Wastes shared research budget

If you need a long session, use Jupyter (closes idle sessions) or batch jobs.

::::::::::::::::::::::::::::::::::::::::::::::::

### 4. Batch Similar Jobs

Instead of:
```bash
# 100 separate 1-minute GPU jobs
for i in {1..100}; do
    sbatch job_$i.sh
done
```

Batch them:
```bash
# One job with 100 iterations
sbatch batch_job.sh
```

## Monitoring Your Usage

Check how many GPUs you're using:

```bash
squeue -u $USER | grep gpu
# Count running jobs with --gres=gpu
```

Check past usage:

```bash
sacct -u $USER --format=JobName,Partition,NodeList,GRES,State,Elapsed
```

## Sharing GPU Nodes

Multiple users may run on same node. Be considerate:
- Don't monopolize node resources
- Monitor your memory usage
- Release GPUs immediately when done
- Don't interfere with others' jobs

## Best Practices Summary

1. **Test with small jobs first**
   - Use L40S, small batch size
   - Run for 15-30 minutes
   - Verify code works
2. **Scale up incrementally**
   - If test works, try larger GPU
   - Increase batch size gradually
   - Monitor memory usage
3. **Clean up after yourself**
   - Delete temporary files
   - Exit interactive sessions
   - Cancel long jobs if not needed
4. **Share responsibility**
   - Leave GPUs for others
   - Don't submit unnecessary jobs
   - Report wasteful jobs you see
5. **Plan ahead**
   - Submit jobs during off-peak (evening, weekends)
   - Peak: business hours, Monday-Friday
   - Expect longer waits during peaks

## Example: Responsible Workflow

```bash
# Step 1: Test on CPU
python3 train.py --gpu false --epochs 1

# Step 2: Quick GPU test
sbatch --gres=gpu:l40s:1 --time=00:30:00 train.sh

# Step 3: If works, scale up
sbatch --gres=gpu:a100:1 --time=04:00:00 train.sh

# Step 4: Monitor and iterate
squeue -u $USER
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Resource Audit

A system administrator shows you the following `squeue` output for a single user:

```
JOBID  PARTITION  NAME          USER    ST  TIME       NODES  GRES          TIME_LIMIT
44201  gpu        interactive   jsmith  R   18:32:05   1      gpu:a100:2    24:00:00
44215  gpu        train_v1      jsmith  R   06:15:22   1      gpu:a100:1    72:00:00
44216  gpu        test_small    jsmith  R   00:02:14   1      gpu:l40s:1    48:00:00
44220  gpu        debug_run     jsmith  R   00:45:30   1      gpu:l40s:1    24:00:00
```

The administrator also notes that `nvidia-smi` on the interactive job (44201) shows
0% GPU utilization, and the `test_small` job (44216) finishes its work in under
5 minutes.

1. Identify **three** wasteful patterns in this user's GPU usage.
2. Suggest a specific improvement for each.

::::::::::::::::::::::::::::::::::::: solution

## Solution

**Wasteful pattern 1: Idle interactive session hogging 2 A100s**

Job 44201 is an interactive session that has been running for over 18 hours with
0% GPU utilization, holding 2 A100 GPUs (the most in-demand resource on the cluster).

**Improvement:** Close interactive sessions when not actively using them. If the user
needs occasional interactive access, request a single L40S for shorter periods:
```bash
srun -p gpu --gres=gpu:l40s:1 --time=02:00:00 --pty bash
```

**Wasteful pattern 2: Massively over-allocated time limits**

Job 44216 (`test_small`) has a 48-hour time limit but finishes in under 5 minutes.
Job 44215 has a 72-hour limit. Over-requesting time blocks the scheduler from
backfilling other users' jobs into that slot.

**Improvement:** Set time limits close to actual need. For a 5-minute test job:
```bash
#SBATCH --time=00:30:00   # 30 minutes is plenty for a quick test
```

**Wasteful pattern 3: Using premium GPUs for small/debug workloads**

Job 44216 uses an L40S (48GB) for a small test, and job 44220 could potentially
be consolidated. Small tests and debugging do not need high-end GPUs.

**Improvement:** Use L40S for testing and debugging, reserve A100 and RTX PRO 6000 for
production training runs that actually need the memory and compute:
```bash
# For testing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00

# Only scale up when the code is verified
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
```

Overall, this user is consuming 4 GPUs (including 3 A100s) simultaneously, most of
which are idle or over-provisioned. Right-sizing would free resources for other
researchers.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints
- Sagehen HPC has limited GPUs; fair sharing is essential
- Request appropriate GPU type for your workload
- Don't monopolize GPUs or leave idle
- Test on L40S before scaling to A100
- Clean up after jobs complete
- Monitor your usage with squeue
- Respect other users' resource needs
::::::::::::::::::::::::::::::::::::::::::::::::
