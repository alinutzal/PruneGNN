#!/usr/bin/bash

input_dir=$1  # main directory where csv files are stored
batch_list=$2 # file listing the subfolders containing the csv (those subfolders whould be in input_dir)
output_dir=$3 # output directory where the Module Maps will be stored

#workdir=/sps/atlas/a/avallier/itk/GNN4ITkTeam/ModuleMapGraph
workdir=${Tracking_path}

while read -r folder ; do
    slurm_log_dir=logs/
    subMMtag=$folder
    mkdir -p $output_dir/"$folder"
    
    sbatch -o "$slurm_log_dir"/ModuleMapCreator.mpi."$folder".log \
        $workdir/slurm/ModuleMapCreator.scheduler.slurm "$workdir" $input_dir/"$folder" $output_dir/"$folder" "$subMMtag" 
done < "$batch_list"
#done < grid.filenames.txt

