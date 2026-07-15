#!/bin/bash
#SBATCH --job-name=cnn_final_holdout
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=./SLURM_suin/R-%SLURM.%j.out

mkdir -p ./SLURM_suin
source /home/sp00001/miniconda3/bin/activate
conda activate tf_env

python -u /home/sp00001/blood_mil_project/suin_scripts/CNN/06b_train_final_and_eval_holdout.py \
    --organized_dir /home/sp00001/blood_mil_project/organized_data \
    --holdout_dir /home/sp00001/blood_mil_project/holdout_data_for_multiclass \
    --model_dir /home/sp00001/blood_mil_project/models/gen2_cnn \
    --pooling mean \
    --instances_per_step 32 \
    --unfreeze_from layer4 \
    --epochs 30

# Run this AFTER run_cnn_cv_mean.sh (all 5 tasks) has finished and
# aggregate_fold_results.py confirms mean-pooling is still the choice.
# This job trains on ALL non-holdout patients (no val split) and evaluates
# on holdout EXACTLY ONCE -- that holdout number becomes the official
# final result, so don't re-run this repeatedly "to see if it improves".
