# Moving to a real drone: what is ready, what is needed, how to test

The intended use: an operator gives one request, a text (what to do, what the target looks like, what is nearby) and a rough
point on the map for each step. The drone does the rest: flies there, searches, picks the target, lands, drops, picks up,
enters or inspects, and returns. Conditions are real: wind, GPS loss and spoofing, birds, an inexact map, a low battery.

## 1. Status

| level | state |
|---|---|
| Mission brain in the simulator (MuJoCo, real OSM maps, faults, wind, birds) | works; results in section 8 |
| Mission brain commanding the real PX4 autopilot firmware (its SIH simulator, same code, MAVLink) | the link works; mission success did not carry over (1 of 12 vs 7 of 12), section 9 |
| Software for a real drone: autopilot link, sensors, "one request and fly" | written (`real/`), not tested on hardware |
| Vision on real photographs | not ready: trained on synthetic frames only, needs fine-tuning on real images |
| Field tests | none |

In short, **the software is ready for bench tests, not for field flights and not for operation.** Before the first flight the
brain must be re-tuned on PX4 (section 9), the drone built (section 2), the detector fine-tuned on real images (section 6),
test stages 0 to 3 passed (section 7) and the local regulations met (section 10). This is roughly TRL 4 (validated in
simulation and with the real firmware); stages 1 to 3 would bring it to TRL 5 or 6.

## 2. Hardware (example set)

| part | what | why | note |
|---|---|---|---|
| frame and motors | quadcopter, 10–15" props, take-off mass 3–4 kg, payload up to 1 kg | 0.3–1 kg loads as in the simulator | 20–30 min with a load |
| autopilot | Pixhawk 6C/6X (PX4 or ArduPilot) | stabilisation, state estimation (EKF2), failsafes | the brain commands the autopilot and does not replace it |
| GNSS | dual-band receiver (u-blox F9P/M9N) and compass | position | a second receiver helps to notice spoofing |
| companion computer | NVIDIA Jetson Orin NX 16 GB (or Orin Nano 8 GB) | vision (YOLO), feeling core, map navigation | 10–25 W, 5–10 % of motor power |
| lidar | 2 × Livox Mid-360 (one upside down) | obstacles, map matching without GPS, hangar doors | as in the simulator; one lidar does not see below −7° |
| down rangefinder | TF-Luna or LightWare | landing, hook rope length | |
| camera | 3-axis gimbal, RTSP, MAVLink Gimbal v2 | search and close check | |
| drop | servo release and a parcel with a parachute on a static line | "drop by parachute" | opens in 0.3–0.6 s |
| pick-up | winch with a hook and a load cell | "pick up the parcel" | weight goes to `HOOK_KG` |
| link | radio telemetry and an LTE modem | questions to the operator with images | without a link the drone never touches an ambiguous target |
| RC transmitter | safety pilot | manual takeover | mandatory |
| drone parachute | recovery system | flights over built-up areas | strongly advised |

## 3. On-board software

```
operator --text+point--> run_real_mission.py --> parser (Qwen2.5-1.5B + LoRA) --> confirmation by the operator
                                   |
                                   v
              mission_brain.Brain (unchanged, the same as in the simulator)
         +-----------+------------+-------------------+--------------+
      nav.Navigator  safety.Safety  perceive.Perceiver  grounding
      (lidar-map)    (feeling core) (YOLO + colour)     (target choice)
                                   |
                         real/real_env.RealEnv  (the simulator's interface)
          +-------------------+-------------------+-------------------+
     real/mav.Link         Ros2Livox            RtspCamera          drop / winch
  (MAVLink: PX4/ArduPilot) (2 lidars, ROS 2)   (video + depth)     (MAVLink commands)
```

- `real/mav.py` keeps the autopilot link: all frame conversions (the brain uses ENU and FLU, the autopilot NED and FRD),
  a 20 Hz velocity setpoint stream, OFFBOARD (PX4) or GUIDED (ArduPilot), arming, gimbal, drop and winch. A watchdog switches
  the stream to hover if the brain sends no command for 0.3 s; if the companion computer itself fails, the autopilot failsafe
  takes over (OFFBOARD loss, then hold or return).
- `real/real_env.py` is the brain's environment on a real drone: autopilot telemetry becomes the same observation as in the
  simulator. Lidars come through ROS 2 (`livox_ros_driver2`), the camera through RTSP; camera depth comes from lidar points,
  and where there are none, from the ground plane measured by the rangefinder.
- `real/run_real_mission.py` runs "one request and fly": the OSM map of the area, parsing, confirmation, flight, questions to
  the operator in the terminal (candidate images are saved to `asks/`), and at the end a landing with motors off or a return by
  the autopilot.
- `real/sitl_cosim.py` checks the brain with the real PX4 firmware (section 9).

## 4. Running on the drone

1. On the Jetson: Ubuntu 22.04 or 24.04, Python ≥ 3.10, `pip install pymavlink torch ultralytics mujoco shapely scipy pillow numpy`
   (MuJoCo only builds the map), ROS 2 and `livox_ros_driver2` for the lidars.
2. Connect the autopilot to the Jetson over UART or USB (MAVLink 2, 921600) or Ethernet; in PX4, `MAV_1_CONFIG` points to
   that port.
3. Set the autopilot parameters (section 5) and pass the bench check (section 7, stage 0).
4. Run:
   `python real/run_real_mission.py --text "Land on the blue pad with the letter H near the five-storey building" --point 55.8231,49.1302 --mavlink serial:/dev/ttyTHS1:921600 --parser models/parser_adapter --base <Qwen2.5-1.5B-Instruct>`
   `--dry` builds only the map, the parse and the points, without connecting to the drone.

## 5. Autopilot settings (PX4): protection that does not depend on the brain

| parameter | value | meaning |
|---|---|---|
| `COM_OF_LOSS_T` | 0.5–1 s | loss of the companion computer's setpoint stream |
| `COM_OBL_RC_ACT` | hold or return | action on OFFBOARD loss |
| `GF_ACTION`, `GF_MAX_HOR_DIST`, `GF_MAX_VER_DIST` | return; 300–500 m; 60–120 m | geofence around the mission area |
| `BAT_LOW_THR`, `BAT_CRIT_THR`, `COM_LOW_BAT_ACT` | 25 %, 15 %, return or land | last battery line (the brain's own rule acts earlier) |
| `RTL_RETURN_ALT` | 10 m above the area's roofs | return altitude |
| `MPC_XY_VEL_MAX`, `MPC_Z_VEL_MAX_DN` | 10 m/s, 1.5 m/s | speed limits |
| `NAV_DLL_ACT` | return | loss of the ground-station link |
| `COM_RC_IN_MODE` | RC required | manual takeover by the pilot |

## 6. What is missing and what may fail in reality

1. The brain on a real autopilot. Time-outs, speeds and precise phases are tuned to the simulator's controller; on PX4 missions
   rarely succeed (section 9). This is the first job before flights and can be done on the bench.
2. Vision. The detector and the colour network are trained on MuJoCo frames; accuracy on real photographs will drop sharply.
   About 2–5 thousand real frames of pads, drop zones, hangars and pick-up points from different heights and weather need boxes
   and colours, then fine-tuning. Threshold as in F1: recall and precision ≥ 0.9.
3. The map. OSM building outlines are off by 1–3 m and heights are often missing. Navigation without GPS relies on walls and
   drifts over open fields (as in the simulator, where the "lost" feeling notices it).
4. Camera depth. The simulator gives exact depth; a real drone has lidar points and the ground plane only. Position errors are
   larger for tall objects (masts, towers); checked at stage 2.
5. Wind near buildings is stronger and gustier than the model; without an airspeed sensor the autopilot estimates wind
   roughly, so drop misses will be larger. Check: parachute drops of a sandbag at stage 2.
6. Gimbal and latency. A real gimbal turns in 0.3–0.5 s and video lags by 0.1–0.3 s. The code pauses before a still; this must
   be checked on the bench.
7. Birds and obstacles. The feeling core is trained in simulation on a Mid-360-class lidar; transfer to a real lidar is not
   tested.
8. Load and hook. Hooking is idealised in the simulator (attach when closer than 0.3 m). A real hook and parcel handle need
   their own design and tests.
9. Parsing is exact on 56 % of hand-written tasks, so the operator's confirmation is mandatory.
10. Repeated questions. In 3 of 182 missions with an operator the drone asked again and again (131–541 times) when no
    candidate was the target. A limit is needed before flights: skip the step after two or three "none" answers.
11. Energy reserve. The learned "low energy" feeling does not prevent a dead battery in the air in hard missions; an explicit
    telemetry rule does better (section 8). On board, this rule must manage the reserve.
12. Lidar noise and untested conditions. Whole-mission tests included GPS loss and spoofing, wind and birds, but not lidar
    faults, fog or a biased compass. A manual flight with spurious lidar points showed the brain climbing without limit and
    losing the map match. The app and `real/run_real_mission.py` now use a lone-point lidar filter, climb only on an obstacle
    seen in three scans in a row, and cap the altitude at 120 m; this is not yet tested on whole missions.
13. Hangars and indoor spaces have no GPS and little room and are tested in simulation only. Entering a hangar remains the
    weakest action (0.14 of steps alone).

## 7. Test stages (the next stage only after the previous one passes)

| stage | where | what | pass criterion |
|---|---|---|---|
| 0. bench (HITL) | Pixhawk and Jetson on a table, PX4 simulator | link, frames, watchdog, RC takeover; re-tuning the brain on PX4 | every command has the right sign, takeover ≤ 0.5 s; on SIH missions succeed no worse than on our controller |
| 1. hover | field, 2–5 m, safety pilot | velocity tracking, OFFBOARD, geofence, return | velocity error ≤ 0.5 m/s, every failsafe triggers |
| 2. field with mock-ups | field, banner "H" pads 2–3 m wide, drop zones (rings on a tarp), a garage or tent as a "hangar", within line of sight | each action 20 times; collecting frames for vision fine-tuning | 0 wrong targets, success no lower than in simulation |
| 3. real places | with permissions, nobody in the area | GPS loss (switching GPS off in the autopilot in flight; **jamming GPS is illegal**), wind, low battery | 0 wrong targets, 0 contacts |
| 4. beyond line of sight | only where legally allowed | long missions | |

**Stop rule:** any wrong target or unexpected contact stops the tests until the cause is found.

## 8. What the final simulator test showed (G2)

300 fresh missions on the held-out places (residential blocks of Tokyo and Kazan, never used in training), with new layouts,
texts, rough points, GPS faults, gusty wind and bird attacks. The scoring rules were written before the run.

| | drone alone | drone with an operator (182 missions) |
|---|---|---|
| steps completed | 0.30 | 0.56 (alone on the same missions: 0.29) |
| whole missions completed | 0.29 | 0.54 |
| missions with a static contact | 4.3 % (without the feeling core 14.7 %) | 10.4 % |
| wrong target | 3 of 300, 1.0 % (upper bound 2.3 %) | 4 of 182, 2.2 % |
| battery dead in the air | 0 | 0 |

By action, alone: fly-by 0.49, landing 0.44, inspection 0.30, hook 0.29, parachute drop 0.22, hangar entry 0.14. With an
operator: fly-by 0.86, landing 0.75, hook 0.65, drop 0.51, inspection 0.39, entry 0.13.

- The feeling core gives the main safety gain: without it, three times more missions touch buildings (14.7 % vs 4.3 %), and
  attacking birds hit more often.
- Alone, the drone is cautious: if the description fits several objects, it touches nothing, so it completes fewer steps.
- With an operator who answers questions it completes almost twice as many steps, at the price of more static contacts.
- The wrong-target safety rule (≤ 2 % at the upper bound) is not met. This is the main risk, so the operator's confirmation of
  the parse and the stop rule in tests are mandatory.

**Hard missions with a weak battery (G3).** 99 missions on the held-out places: 30–65 % charge at take-off, a cold battery,
wind up to 9 m/s, GPS loss or spoofing, birds, return required. With the learned slow feelings the battery died in the air in
29 % of missions; with a simple telemetry rule (is the charge enough to fly home at the current consumption) in 16 %, and the
rule brought the drone home more often (63 % vs 52 %). The "low energy" feeling was learned on short episodes and does not know
the distance home. On a real drone, an explicit rule "consumption × distance home + reserve" must manage energy, with the
autopilot battery failsafe (section 5) as the last line.

## 9. The brain with the real PX4 firmware (G5)

The MAVLink bridge (`real/mav.py`, `real/sitl_cosim.py`) connected the unchanged brain to PX4 v1.16 in its own quadcopter
simulator SIH. Every 0.1 s our world was placed at the position estimated by PX4, so that the lidar, camera and scoring stayed
ours. 12 missions: landing, drop, pick-up, inspection, fly-by, targets on the ground.

- The mechanics work. PX4 armed, took off, followed the brain's commands, landed, re-armed, and refused to take off when its
  sensors failed. The watchdog triggered when the brain was late.
- Mission success did not carry over: 1 of 12 with PX4, 7 of 12 with our controller on the same missions. Holding positions with
  PX4's own position loop in the precise phases changed nothing (also 1 of 12).
- Cause: PX4 tracks the brain's velocity commands with a typical error of 0.8–1.3 m/s and missions take about twice as long. The
  brain is tuned to the fast, precise controller of our simulator, and with PX4 its time-outs and precise phases (approach,
  close check, landing, hooking) no longer match how the aircraft flies.

Before flights: re-tune the brain's time-outs, speeds and precise phases on PX4 (bench, stage 0), use PX4 position setpoints
with a smooth approach in the precise phases, and repeat G5 on at least 50 missions. Criterion: on SIH no worse than our
controller and 0 wrong targets.

## 10. Regulations

Rules for unmanned aircraft differ by country and region and change often. Before any flight, check at least: registration of
the aircraft, permission to use the airspace and the flight plan, local bans and restricted zones (airports, protected sites),
the legal regime for flights beyond line of sight and for cargo delivery, the ban on dropping loads over people, and liability
insurance. This list is not legal advice.

## 11. Civil scope

Targets are places and structures only: pads, drop zones, pick-up points, hangars, masts, towers. People and vehicles are never
targets and are not recognised. If something moves on a drop zone (by lidar), the drone waits or cancels the drop.
