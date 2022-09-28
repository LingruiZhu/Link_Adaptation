#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 02-05:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=olla_simulation
#SBATCH -o slurm_message/outputs/olla_simulation.out
#SBATCH -e slurm_message/errors/olla_sumulation.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 olla_simulation.py
wait
echo “$(date): Finished jobs!”