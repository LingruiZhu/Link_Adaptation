#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 00-01:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=mcs_bler_simulation
#SBATCH -o slurm_message/outputs/mcs_bler_simulation.out
#SBATCH -e slurm_message/errors/mcs_bler_simulation.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 mcs_bler_simulation.py
wait
echo “$(date): Finished jobs!”