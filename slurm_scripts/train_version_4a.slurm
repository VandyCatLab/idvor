#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constrain=haswell 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00

#SBATCH --array=10-12
#SBATCH --output=../logs/Version_4/train_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Seeds 3, 5 and 9 crapped out early. Try 3 new seeds.
singularity exec ../project_baseline.simg python ../python_scripts/train.py ${SLURM_ARRAY_TASK_ID}
