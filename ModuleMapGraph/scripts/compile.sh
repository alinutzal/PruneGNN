#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

PS3='please choose a compiler: '
compilers=("g++" "g++ & gprof" "intel icpc" "mpic++" "nvcc" "nvc++" "Clean" "Return to main menu")
select opt in "${compilers[@]}"; do
  case $opt in
    "g++")
      source $Tracking_path/CPU/scripts/compile.g++.sh
      ;;
    "g++ & gprof")
      source $Tracking_path/CPU/scripts/compile.g++.gprof.sh
      ;;
    "intel icpc")
      source $Tracking_path/CPU/scripts/compile.intel.sh
      ;;
    "mpic++")
      source $Tracking_path/MPI/scripts/compile.mpi.sh
      ;;
    "nvcc")
      source $Tracking_path/GPU/scripts/compile.nvcc.sh
      ;;
    "nvc++")
      source $Tracking_path/GPU/scripts/compile.nvc++.sh
      ;;
    "Clean")
      source $Tracking_path/scripts/clean.sh
      ;;
    "Return to main menu")
      echo -e "\033[34mMain Menu\033[0m"
      break
      ;;
    *) echo -e "\033[31minvalid option\033[0m $REPLY"
      ;;
  esac
  REPLY=
done
