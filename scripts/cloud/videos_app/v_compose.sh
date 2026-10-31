#!/usr/bin/env bash
# Compose the side-by-side videos again (after v.sh has finished) with localised captions. Marker COMPOSE_DONE.
set -u; A=$HOME/af08; O=$A/runs/videos_v3; . $HOME/venv/bin/activate; cd $A/src
while [ ! -f $A/V_DONE ]; do sleep 20; done
for i in 42 45 112 206; do
  F=$A/runs/vr/G${i}_full; R=$A/runs/vr/G${i}_reflex
  for LG in ru en; do
    timeout 1800 python compose_compare.py $F $R $O/G${i}_compare_$LG.mp4 $LG 25 >> $A/runs/compose2.log 2>&1
    python compose_compare.py $F $R x $LG --still $O/G${i}_compare_$LG.jpg >> $A/runs/compose2.log 2>&1
  done
done
cd $A/runs && tar czf v.tgz *.log check_v.json videos_v3 && touch $A/COMPOSE_DONE
