ModuleMapCreator.exe \
    --input-dir=CI/CPU/data.$year/events$year-1 \
    --input-filename-pattern=event \
    --output-module-map=$Tracking_path/log/ModuleMap.$year.10evts.1 \
    --output-dir=OUTPUT_DIR_GEN \
    --give-true-graph=True \
    --save-graph-on-disk-graphml=True \
    --save-graph-on-disk-npz=True \
    --min-pt-cut=1. \
    --min-nhits=3 \
    --phi-slice=False \
    --cut1-phi-slice=0. \
    --cut2-phi-slice=0. \
    --eta-region=False \
    --cut1-eta=0. \
    --cut2-eta=0. \

