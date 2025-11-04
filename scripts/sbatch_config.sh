#!/bin/sh
#SBATCH --nodes=2
#SBATCH --cpus-per-task=128
#SBATCH --gpus=16
#SBATCH --output=test-slurm.out
#SBATCH --mem=512000
#SBATCH --exclude=cn[8]

export GPUS_PER_NODE=8
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER="
    accelerate launch \
    --config_file src/configs/fsdp_config.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

export SCRIPT="src/training/train_model.py --config src/configs/train_qwen_aggresive.nip"

export CMD="$LAUNCHER $SCRIPT"

echo "$head_node_ip"
echo "$SLURM_JOB_NODELIST"
echo "$CMD"

srun --container-image /scratch/d.melikhov/images/dmitry315_learn_env_6.sqsh --container-writable \
    --container-mounts /scratch/d.melikhov/gitlab_projects/ELlama/:/app/ \
    --container-workdir /app bash -c "ls -la && ln -s /opt/conda/bin/python /usr/bin/python3 && pip install nip-config && pip install -e . && $CMD" 1>stdout_full.log 2>stderr_full.log