#!/usr/bin/env bash
# PX4 SITL with the SIH model: the real PX4 flight stack (EKF2, position / velocity controllers, offboard mode, failsafes) runs on
# the VM and our brain flies it over MAVLink (REAL_DRONE_RU.md, check G5). Not a hypothesis: a readiness check. Low priority (nice):
# the stage G runs come first.
set -u; A=$HOME/af08; L=$A/runs/setup.log; X=$HOME/px4; echo "px4 start $(date)" >> $L
for TAG in v1.16.0 v1.15.4; do
  rm -rf $X; timeout 1800 git clone --depth 1 -b $TAG --recursive --shallow-submodules -j 8 https://github.com/PX4/PX4-Autopilot.git $X >> $A/runs/px4_clone.log 2>&1 && break
done
echo "px4 clone rc=$? tag $TAG $(du -sh $X 2>/dev/null | cut -f1) $(date)" >> $L
python3 -m venv $HOME/px4venv; . $HOME/px4venv/bin/activate
MIR="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
timeout 1200 pip install -q --timeout 30 $MIR empy==3.3.4 jinja2 pyyaml kconfiglib pyros-genmsg packaging toml numpy jsonschema future pyserial lxml >> $A/runs/px4_pip.log 2>&1 || \
  timeout 1200 pip install -q --timeout 30 empy==3.3.4 jinja2 pyyaml kconfiglib pyros-genmsg packaging toml numpy jsonschema future pyserial lxml >> $A/runs/px4_pip.log 2>&1
echo "px4 pip rc=$? $(date)" >> $L
cd $X && nice -n 15 timeout 3600 make px4_sitl_default >> $A/runs/px4_make.log 2>&1; echo "px4 build rc=$? $(date)" >> $L
ls $X/build/px4_sitl_default/bin/ >> $L 2>&1; ls $X/ROMFS/px4fmu_common/init.d-posix/airframes/ 2>/dev/null | grep -i sih >> $L
touch $A/PX4_BUILT
