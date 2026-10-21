#!/bin/zsh
# Render S2-F sets A then B, two chunks of 10 in parallel each (same settings as s2-render).
P=/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021
R=/Volumes/harddisk/fastfillout/unknownsize-20261013/code-snapshots/main-e6992cbde/fastfill/comparison/render_assets.py
for S in A B; do
  for i in 0 1; do
    caffeinate -dimsu /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python $R -- \
      --manifest $P/s2f/manifest-$S-$i.json --output $P/s2f/render-$S-$i --resolution 512 --samples 32 --top --denoise --threads 3 \
      > $P/s2f/render-$S-$i.log 2>&1 &
  done
  wait
done
tail -n 1 $P/s2f/render-*.log
