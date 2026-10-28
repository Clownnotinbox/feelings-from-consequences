#!/usr/bin/env bash
# The app on the video VM (after v.sh has set up the venv). usage: v_app.sh test|walk [ru|en]
#   test: the end-to-end browser check (runs/apptest_LANG/app_test.json + screenshots), marker APP_TEST_DONE_LANG
#   walk: the walkthrough recording and its captioned video (runs/videos_v3/walkthrough_LANG.mp4), marker APP_DONE_LANG
set -u; A=$HOME/af08; L=$A/runs/v_app.log; O=$A/runs/videos_v3; LG=${2:-ru}; mkdir -p $O
export OSM2_DIR=$A/osm2 MUJOCO_GL=egl YOLO_OFFLINE=1 PYTHONIOENCODING=utf-8; . $HOME/venv/bin/activate; cd $A/src
pkill -f "[d]emo_server.py"; sleep 1
(OMP_NUM_THREADS=2 python demo_server.py --place kazan_novosav --seed 7 --port 8765 --no-browser > $A/runs/app_$1_$LG.log 2>&1 &)
for k in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8765/api/state > /dev/null && break; sleep 5; done
echo "$1 $LG: app up after $((k * 5)) s $(date)" >> $L
if [ "$1" = test ]; then
  rm -rf $A/runs/apptest_$LG; timeout 1500 python app_test.py http://127.0.0.1:8765 $A/runs/apptest_$LG $LG > $A/runs/app_test_$LG.log 2>&1; echo "app_test $LG rc=$? $(date)" >> $L
  pkill -f "[d]emo_server.py"; cd $A/runs && tar czf apptest_$LG.tgz apptest_$LG app_test_$LG.log app_$1_$LG.log 2>/dev/null; touch $A/APP_TEST_DONE_$LG
else
  W=$A/runs/walk_$LG; rm -rf $W; timeout 1800 python record_walkthrough.py http://127.0.0.1:8765 $W 1 $LG > $A/runs/walk_$LG.log 2>&1; echo "walk $LG rc=$? $(date)" >> $L
  pkill -f "[d]emo_server.py"
  timeout 1200 python compose_walk_video.py $W $O/walkthrough_$LG.mp4 $LG >> $A/runs/compose.log 2>&1
  echo "walk video $LG $(ls -la $O/walkthrough_$LG.mp4) $(date)" >> $L; touch $A/APP_DONE_$LG
fi
