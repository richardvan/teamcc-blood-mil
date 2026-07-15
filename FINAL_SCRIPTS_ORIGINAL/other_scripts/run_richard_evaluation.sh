#! /bin/bash

#SBATCH --job-name=run_richard_eval
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]
#SBATCH --output=./SLURM_richard/E-%SLURM.%j.out

source /home/sp00001/miniconda3/bin/activate


conda activate richard_conda_env
# conda activate project1




# ── Diagnostic block ─────────────────────────────────────────
echo "====== JOB START: $(date) ======"
echo "Node:        $(hostname)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Partition:   $SLURM_JOB_PARTITION"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"   # blank = no GPU allocated!

# nvidia-smi                                     # full GPU status snapshot
echo "==============================="
# ─────────────────────────────────────────────────────────────


python -u /home/sp00001/blood_mil_project/richard_scripts/j_SVM_binary_holdout_eval.py
python -u /home/sp00001/blood_mil_project/richard_scripts/j_SVM_multiclass_holdout_eval.py
python -u /home/sp00001/blood_mil_project/richard_scripts/j_uppercase_MISVM_multiclass_holdout_eval.py
python -u /home/sp00001/blood_mil_project/richard_scripts/j_lowercase_miSVM_multiclass_holdout_eval.py


# name that will show up in the queue(two dashes)
# the partitions to run in GPU(two dashes)
# node in the GPU partition(two dashes)

# directory of /miniconda3/bin/activate
# activate your virtual environment

# run your python code file(two dashes)




echo "====== JOB END: $(date) ======"
