#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

ln -sf $Tracking_path/build/mpi/MPI bin 
mv bin $Tracking_path/build/
PS3='please choose a program to execute: '
programs=("Module Map Creator" "Module Map Scheduler (slurm)" "Module Map Reschedule (slurm)" "Return to main menu")
select opt in "${programs[@]}"; do
  case $opt in
    "Module Map Creator")
      source $Tracking_path/MPI/scripts/ModuleMapCreator.mpi.sh
      ;;
    "Module Map Scheduler (slurm)")
      source $Tracking_path/MPI/scripts/ModuleMapScheduler.hpss.sh
      ;;
    "Module Map Reschedule (slurm)")
      source $Tracking_path/MPI/scripts/ModuleMapReschedule.sh
      ;;
    "Return to main menu")
      echo -e "\033[34mMain Menu exit\033[0m"
      break
      ;;
    *) echo -e "\033[31minvalid option\033[0m $REPLY"
      ;;
  esac
  REPLY=
done
