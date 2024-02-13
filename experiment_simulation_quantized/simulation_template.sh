#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 02-05:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=80000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=reward_normalized_01
#SBATCH -o outputs/reward_normalized_01.out
#SBATCH -e errors/reward_normalized_01.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 simulation_template.py
wait
echo “$(date): Finished jobs!”