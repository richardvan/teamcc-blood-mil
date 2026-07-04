#!/bin/bash
#SBATCH --job-name=mil_model_richard_eval
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]
#SBATCH --output=./SLURM_richard/E-%SLURM.%j.out

source /home/sp00001/miniconda3/bin/activate
conda activate project1

# --- gen1 base model (misvm_v1) ---
python -u /home/sp00001/blood_mil_project/richard_scripts/predict-holdout-MI-SVM.py \
    --model-path /home/sp00001/blood_mil_project/models/gen1_svm/misvm_v1.joblib \
    --model-gen gen1_svm --model-name misvm_v1
python -u /home/sp00001/blood_mil_project/richard_scripts/predict-external-test-MI-SVM.py \
    --model-path /home/sp00001/blood_mil_project/models/gen1_svm/misvm_v1.joblib \
    --model-gen gen1_svm --model-name misvm_v1

# --- gen1 CV-tuned model (misvm_cv_v1) ---
python -u /home/sp00001/blood_mil_project/richard_scripts/predict-holdout-MI-SVM.py \
    --model-path /home/sp00001/blood_mil_project/models/gen1_svm/misvm_cv_v1.joblib \
    --model-gen gen1_svm --model-name misvm_cv_v1
python -u /home/sp00001/blood_mil_project/richard_scripts/predict-external-test-MI-SVM.py \
    --model-path /home/sp00001/blood_mil_project/models/gen1_svm/misvm_cv_v1.joblib \
    --model-gen gen1_svm --model-name misvm_cv_v1
