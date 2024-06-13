#!/bin/bash
#SBATCH -Jtest
#SBATCH --account=gts-dmolzahn6
#SBATCH -N1 --ntasks-per-node=1
#SBATCH -qinferno
#SBATCH --mem-per-cpu=100G
#SBATCH -t100:00:00
#SBATCH -oReport_generate_data%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xcao306@gatech.edu

cd $SLURM_SUBMIT_DIR

module load julia/1.9.2

julia $HOME/LET_GO/vanilla.jl
