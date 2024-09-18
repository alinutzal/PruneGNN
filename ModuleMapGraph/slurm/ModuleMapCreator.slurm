#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2022,2023 ROUGIER Charline
#* copyright © 2022,2023 COLLARD Christophe
#* copyright © 2022,2023 Centre National de la Recherche Scientifique
#* copyright © 2022,2023 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

#
#SBATCH --job-name=ModuleMapCreator
#SBATCH --output=log/ModuleMapCreator.log
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=flash

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
echo -n 'ModuleMapCreator executed on : '
hostname
echo -n 'started on : '
date
echo ------------------------------------------------------

cd $PBS_O_WORKDIR
echo 'Tracking directory: ' $1
cd $1

#execute program
#sh "scripts/ModuleMapCreator.slurm.sh" // recursive call - do not use
sh "MPI/scripts/ModuleMapCreator.mpi.sh"

echo
echo ------------------------------------------------------
#result in file netpipe.o$PBS_JOBID
echo -n 'ended on : '
date
echo ------------------------------------------------------
