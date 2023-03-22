#!/bin/bash
#SBATCH --job-name=ablenerf
#SBATCH --partition=regular
#SBATCH --time=144:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --output="%j.out"

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((11211+$RANDOM))
export WORLD_SIZE=8

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo ${master_addr}
export MASTER_ADDR=$master_addr
### init virtual environment if needed
srun python main.py \
+dataset=drums \
++dataset.factor=0 \


