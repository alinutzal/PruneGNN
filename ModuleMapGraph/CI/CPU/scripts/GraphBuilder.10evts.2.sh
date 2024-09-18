GraphBuilder.exe \
    --input-dir=$Tracking_path/CI/CPU/data.$year/events$year-2 \
    --input-filename-pattern=event \
    --input-module-map=$Tracking_path/CI/CPU/MM2023/ModuleMap.90k \
    --strip-hit-pair=False \
    --output-dir=$Tracking_path/log \
    --give-true-graph=True \
    --save-graph-on-disk-graphml=True \
    --save-graph-on-disk-npz=True \
    --save-graph-on-disk-pyg=True \
    --save-graph-on-disk-csv=True \
    --min-pt-cut=1 \
    --min-nhits=3 \
    --phi-slice=False \
    --cut1-phi-slice=0. \
    --cut2-phi-slice=0. \
    --eta-region=False \
    --cut1-eta=0. \
    --cut2-eta=0. \

