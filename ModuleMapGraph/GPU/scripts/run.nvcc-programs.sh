#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

source $Tracking_path/configure.RTX8000.sh
ln -sf $Tracking_path/build/nvcc/bin bin
mv bin $Tracking_path/build/
PS3='please choose a program to execute: '
programs=("Graph Builder" "Graph Builder (slurm)" "Walk Through" "Return to main menu")
select opt in "${programs[@]}"; do
  case $opt in
    "Graph Builder")
      source $Tracking_path/scripts/GraphBuilder.sh
      ;;
    "Graph Builder (slurm)")
      source $Tracking_path/GPU/scripts/GraphBuilder.slurm.sh
      ;;
    "Walk Through")
      source $Tracking_path/GPU/scripts/WalkThrough.sh
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
