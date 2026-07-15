#!/bin/bash
#SBATCH --job-name=cnn_cv_mean
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=1-5
#SBATCH --output=./SLURM_suin/R-%SLURM.%A_%a.out

mkdir -p ./SLURM_suin
source /home/sp00001/miniconda3/bin/activate
conda activate tf_env

python -u /home/sp00001/blood_mil_project/suin_scripts/CNN/06a_train_cnn_cv.py \
    --organized_dir /home/sp00001/blood_mil_project/organized_data \
    --cv_dir /home/sp00001/blood_mil_project/cv_splits_for_multiclass \
    --model_dir /home/sp00001/blood_mil_project/models/gen2_cnn \
    --fold "$SLURM_ARRAY_TASK_ID" \
    --pooling mean \
    --instances_per_step 32 \
    --unfreeze_from layer4 \
    --max_epochs 500 \
    --patience 15

# --array=1-5 : one task per fold, mean-pooling only (max/min_max already
#               ruled out via the earlier 5-fold comparison).
