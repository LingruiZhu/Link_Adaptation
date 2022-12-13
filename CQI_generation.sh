#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 02-05:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=CQI_gen
#SBATCH -o slurm_message/outputs/CQI_gen.out
#SBATCH -e slurm_message/errors/CQI_gen.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 CQI_generation.py
wait
echo “$(date): Finished jobs!”