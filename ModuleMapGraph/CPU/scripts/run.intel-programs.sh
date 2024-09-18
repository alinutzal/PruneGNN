#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

ln -sf $Tracking_path/build/intel/CPU bin
mv bin $Tracking_path/build/
PS3='please choose a program to execute: '
programs=("Module Map Creator" "Module Map Creator (slurm)" "Module Map File Merger" "Graph Builder" "Graph Builder (slurm)" "Root Converter" "Return to main menu")
select opt in "${programs[@]}"; do
  case $opt in
    "Module Map Creator")
      source $Tracking_path/CPU/scripts/ModuleMapCreator.sh
      ;;
    "Module Map Creator (slurm)")
      source $Tracking_path/CPU/scripts/ModuleMapCreator.slurm.sh
      ;;
    "Module Map File Merger")
      source $Tracking_path/MPI/scripts/ModuleMapFileMerger.sh
      ;;
    "Graph Builder")
      source $Tracking_path/scripts/GraphBuilder.sh
      ;;
    "Graph Builder (slurm)")
      source $Tracking_path/CPU/scripts/GraphBuilder.slurm.sh
      ;;
    "Root Converter")
      source $Tracking_path/CPU/scripts/root_converter.sh
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
