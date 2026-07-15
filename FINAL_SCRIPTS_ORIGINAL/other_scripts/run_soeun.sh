#!/bin/bash
#SBATCH --job-name=mil_soeun
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --output=./SLURM_soeun/R-%SLURM.%j.out

source /home/sp00001/miniconda3/bin/activate
conda activate tf_env
# run_soeun.sh
cd /home/sp00001/blood_mil_project/soeun_scripts
python 07_6_multi_attention_mil_train.py --epochs 60

# name that will show up in the queue(two dashes)
# the partitions to run in GPU(two dashes)
# node in the GPU partition(two dashes)

# directory of /miniconda3/bin/activate
# activate your virtual environment

# run your python code file(two dashes)
