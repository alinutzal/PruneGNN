#!/bin/bash
#***************************************************************
#* Tracking project library - L2IT
#* Trace reconstruction in LHC
#* copyright © 2024 COLLARD Christophe
#* copyright © 2024 Centre National de la Recherche Scientifique
#* copyright © 2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
#***************************************************************/

mkdir $Tracking_path/log
ln -sf $Tracking_path/build/g++/bin $Tracking_path/build
PS3='please choose a program to profile: '
programs=("Module Map Creator" "Module Map File Merger" "Graph Builder" "Graph Builder (GPU nvprof)" "Graph Builder (GPU nvc)" "Graph Builder (GPU nsys)" "Root Converter" "Return to main menu")
select opt in "${programs[@]}"; do
  case $opt in
    "Module Map Creator")
      gprof source $Tracking_path/scripts/ModuleMapCreator.sh > ModuleMapCreator.gprof
      echo -e log written in file ModuleMapCreator.gprof
      ;;
    "Module Map File Merger")
      gprof source $Tracking_path/MPI/scripts/ModuleMapFileMerger.sh > ModuleMapFileMerger.gprof
      echo -e log written in file ModuleMapFile Merger.gprof
      ;;
    "Graph Builder")
      gprof $Tracking_path/build/bin/GraphBuilder.exe > GraphBuilder.gprof
      echo -e log written in file GraphBuilder.gprof
      ;;
    "Graph Builder (GPU nvprof)")
      ln -sf $Tracking_path/build/nvcc/bin $Tracking_path/build/
      /usr/local/cuda-12.1/bin/nvprof ./build/bin/GraphBuilder.exe     --input-dir=1event     --input-module-map=ModuleMap.90k     --input-strip-modules=stripModules_v00_2023-06-07.root     --strip-hit-pair=False     --output-dir=OUTPUT_DIR_GEN     --give-true-graph=True     --save-graph-on-disk-graphml=True     --save-graph-on-disk-npz=True     --save-graph-on-disk-pyg=True     --save-graph-on-disk-csv=True     --min-pt-cut=1     --min-nhits=3     --phi-slice=False     --cut1-phi-slice=0.     --cut2-phi-slice=0.     --eta-region=False     --cut1-eta=0.     --cut2-eta=0.
      ;;
    "Graph Builder (GPU nvc)")
      ln -sf $Tracking_path/build/nvcc/bin $Tracking_path/build/
      /usr/local/cuda-12.1/bin/ncu ./build/bin/GraphBuilder.exe --input-dir=1event --input-module-map=ModuleMap.90k --input-strip-modules=stripModules_v00_2023-06-07.root --strip-hit-pair=False --output-dir=OUTPUT_DIR_GEN --give-true-graph=True --save-graph-on-disk-graphml=True --save-graph-on-disk-npz=True --save-graph-on-disk-pyg=True --save-graph-on-disk-csv=True --min-pt-cut=1 --min-nhits=3 --phi-slice=False --cut1-phi-slice=0. --cut2-phi-slice=0. --eta-region=False --cut1-eta=0. --cut2-eta=0.
      ;;
   "Graph Builder (GPU nsys)")
      ln -sf $Tracking_path/build/nvcc/bin $Tracking_path/build/
      /usr/local/cuda-12.1/bin/nsys profile ./build/bin/GraphBuilder.exe --input-dir=1event --input-module-map=ModuleMap.90k --input-strip-modules=stripModules_v00_2023-06-07.root --strip-hit-pair=False --output-dir=OUTPUT_DIR_GEN --give-true-graph=True --save-graph-on-disk-graphml=True --save-graph-on-disk-npz=True --save-graph-on-disk-pyg=True --save-graph-on-disk-csv=True --min-pt-cut=1 --min-nhits=3 --phi-slice=False --cut1-phi-slice=0. --cut2-phi-slice=0. --eta-region=False --cut1-eta=0. --cut2-eta=0.
      ;;
    "Root Converter")
      gprof $Tracking_path/GPU/scripts/root_converter.sh > RootConverter.gprof
      echo -e log written in file RootConverter.gprof
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
