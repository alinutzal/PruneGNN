
cd $2/..
rm -Rf OUTPUT_DIR_GEN
mkdir OUTPUT_DIR_GEN
cd ..
mpirun -np 6 ModuleMapCreator.mpi.exe \
    --input-dir=$2 \
    --output-module-map=$3/ModuleMap.$4.root \
    --min-pt-cut=1. \
    --min-nhits=3 \
    --log=$3 