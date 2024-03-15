#!/bin/bash
#SBATCH -Jtest
#SBATCH --account=gts-dmolzahn6
#SBATCH -N1 --ntasks-per-node=1
#SBATCH -qinferno
#SBATCH --mem-per-cpu=6G
#SBATCH -t1:00:00
#SBATCH -oReport_generate_data.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xcao306@gatech.edu

cd $SLURM_SUBMIT_DIR

module load gcc/10.3.0-o57x6h
module load julia/1.9.2

julia /storage/home/hcoda1/2/usename/path-to-file
# julia $HOME/LET_GO/vanilla.jl