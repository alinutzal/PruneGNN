#!/usr/bin/bash

# SLURM options:

#SBATCH --job-name=GraphCreator    # Job name
#SBATCH --output=GraphCreator_%j.log   # Standard output and error log

#SBATCH --licenses=sps
#SBATCH --account=atlas
#SBATCH --partition htc               # Partition choice
#SBATCH --ntasks 1                   # Run a single task (by default tasks == CPU)
#SBATCH --time 0-1:00:00             # 7 days by default on htc partition
#SBATCH --mem=16000                   # M?moire en MB par d?faut
#SBATCH --mail-user=alexis.vallier@l2it.in2p3.fr   # Where to send mail
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Setup the env
source setup_cvmfs_lcg103.sh

./GraphBuilder.exe \
    --input-dir=$1 \
    --input-module-map=/sps/l2it/CommonData/MM_23/MMtriplet_1GeV_3hits_noE__merged__sorted_converted.root \
    --output-dir=$2 \
    --give-true-graph=True \
    --save-graph-on-disk-graphml=False \
    --save-graph-on-disk-npz=True \
    --min-pt-cut=1 \
    --min-nhits=3 \
    --phi-slice=False \
    --eta-region=False \
