#! /bin/bash

#SBATCH --job-name=mil_model
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]
#SBATCH --output=./SLURM_jihye/R-%SLURM.%j.out

source /home/sp00001/miniconda3/bin/activate

conda activate project1

python -u /home/sp00001/05_svm_mil_multiclass.py \
    --organized_dir /home/sp00001/blood_mil_project/organized_data \
    --cv_dir /home/sp00001/blood_mil_project/cv_splits \
    --n_folds 5 \
    --max_cells_per_patient 0


# name that will show up in the queue(two dashes)
# the partitions to run in GPU(two dashes)
# node in the GPU partition(two dashes)

# directory of /miniconda3/bin/activate
# activate your virtual environment

# run your python code file(two dashes)
