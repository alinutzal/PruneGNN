#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

PS3='please choose a compiler: '
compilers=("clean g++ build" "clean intel build" "clean mpi build" "clean nvcc build" "clean nvc++ build" "clean all" "Return to main menu")
select opt in "${compilers[@]}"; do
  case $opt in
    "clean g++ build")
        rm -Rf $Tracking_path/build/g++
        ;;
    "clean intel build")
        rm -Rf $Tracking_path/build/intel
        ;;
    "clean mpi build")
        rm -Rf $Tracking_path/build/mpi
        ;;
    "clean nvcc build")
        rm -Rf $Tracking_path/build/nvcc
        ;;
    "clean nvc++ build")
        rm -Rf $Tracking_path/build/nvc++
        ;;
    "clean all")
      echo -e '\033[31m'
      read -p "Are you sure you want to erase all built versions ?" -n 1 -r
      echo -e '\033[0m'   # (optional) move to a new line
      if [[ $REPLY =~ ^[Yy]$ ]] 
      then
        rm -Rf $Tracking_path/build
      fi
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
