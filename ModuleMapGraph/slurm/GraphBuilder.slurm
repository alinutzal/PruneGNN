#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2023 COLLARD Christophe
#* copyright © 2023 Centre National de la Recherche Scientifique
#* copyright © 2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

#
#SBATCH --job-name=GraphBuilder
#SBATCH --output=log/GraphBuilder.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00
#SBATCH --mem-per-cpu=2500
#SBATCH --partition=hpc

#Affiche infos
NCPU=$SLURM_NTASKS
echo ------------------------------------------------------
echo 'This job allocates ' ${NCPU} ' cpu(s)'
echo 'This job runs on the following node(s) : '
echo $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo 'SLURM : slurm runs on : ' $SLURM_CLUSTER_NAME
echo 'SLURM : job submitted on : '$SLURM_SUBMIT_HOST
echo 'SLURM : queue : ' $SLURM_JOB_PARTITION
echo 'SLURM : job id : ' $SLURM_JOB_ID
echo 'SLURM : job name : ' $SLURM_JOB_NAME
echo 'SLURM : working directory : ' $SLURM_SUBMIT_DIR
echo
echo ------------------------------------------------------
echo -n 'GraphBuilder executed on : '
hostname
echo -n 'started on : '
date
echo ------------------------------------------------------

cd $PBS_O_WORKDIR
echo 'Tracking directory: ' $1
cd $1

#execute program
sh "scripts/GraphBuilder.sh"

echo
echo ------------------------------------------------------
#result in file netpipe.o$PBS_JOBID
echo -n 'ended on : '
date
echo ------------------------------------------------------
