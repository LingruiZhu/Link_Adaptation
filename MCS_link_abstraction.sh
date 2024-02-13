#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 20-00:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=80000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=sim_bler_e2e
#SBATCH -o slurm_message/outputs/sim_bler_e2e.out
#SBATCH -e slurm_message/errors/sim_bler_e2e.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 MCS_link_abstraction.py
wait
echo “$(date): Finished jobs!” 