#!/bin/zsh
# Render the 20 S2-F-task picks, two chunks of 10 in parallel (same Blender command and settings as s2f_render.sh).
P=/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021
R=/Volumes/harddisk/fastfillout/unknownsize-20261013/code-snapshots/main-e6992cbde/fastfill/comparison/render_assets.py
for i in 0 1; do
  caffeinate -dimsu /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python $R -- \
    --manifest $P/s2f-task/manifest-$i.json --output $P/s2f-task/render-$i --resolution 512 --samples 32 --top --denoise --threads 3 \
    > $P/s2f-task/render-$i.log 2>&1 &
done
wait
tail -n 1 $P/s2f-task/render-*.log
