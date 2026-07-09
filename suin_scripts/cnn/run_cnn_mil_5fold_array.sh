#!/bin/bash
#SBATCH --job-name=cnn_mil_5fold
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=./SLURM_suin/R-%SLURM.%A_%a.out

mkdir -p ./SLURM_suin

# No --nodelist -- let Slurm pick a free GPU across the partition.

source /home/sp00001/miniconda3/bin/activate
conda activate tf_env

# 15 combos: index = pooling_idx * 5 + (fold - 1)
POOLINGS=(mean max min_max)
FOLDS=(1 2 3 4 5)

POOLING_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
POOLING=${POOLINGS[$POOLING_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

SCRIPT="/home/sp00001/blood_mil_project/suin_scripts/06_cnn_finetune_mil_${POOLING}.py"

echo "Array task $SLURM_ARRAY_TASK_ID -> pooling=$POOLING fold=$FOLD -> $SCRIPT"

python -u "$SCRIPT" \
    --organized_dir /home/sp00001/blood_mil_project/organized_data \
    --cv_dir /home/sp00001/blood_mil_project/cv_splits_for_multiclass \
    --holdout_dir /home/sp00001/blood_mil_project/holdout_data_for_multiclass \
    --model_dir /home/sp00001/blood_mil_project/models/gen2_cnn \
    --fold "$FOLD" \
    --instances_per_step 32 \
    --unfreeze_from layer4 \
    --epochs 30

# --array=0-14 : 15 tasks (3 poolings x 5 folds). min-pooling excluded --
#                already confirmed structurally broken (ReLU features collapse
#                min to ~0), no point spending GPU time re-confirming it.
# Each task writes its own holdout_report_fold{FOLD}_{POOLING}.json to
# models/gen2_cnn/ -- after all 15 finish, run select_best_fold.py to pick,
# per pooling, the fold with the best VALIDATION balanced accuracy (not
# holdout -- see that script's docstring for why this matters).
