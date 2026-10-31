# The app: one request, the drone does the rest

The app flies the simulated drone of the paper in a browser page. You give one request: a click on the map near the target and
a text in your own words saying what to do and what the target looks like. The point is rough; a miss of tens of metres is
fine. The drone takes off, flies over the roofs, searches, checks the target up close and acts: lands, drops a load by
parachute, picks up a parcel with a hook, enters a hangar, inspects or flies by, then returns home. If the description fits
several objects, it shows images of the candidates and asks which one; without an answer it does not touch an ambiguous target.
The interface is in English and Russian (switch in the header).

## Installation (Windows, NVIDIA GPU, Python 3.11)

1. Download `models.zip` from [release v1.0](https://github.com/Clownnotinbox/feelings-from-consequences/releases/tag/v1.0)
   and unpack it in the repository root, so that `models/` and `runs/` sit next to `src/`.
2. Run `install_app.bat` once. It creates `.venv`, installs the packages and downloads the base parser model
   Qwen2.5-1.5B-Instruct (about 3 GB).
3. Run `run_app.bat`. The page opens at `http://localhost:8765`; the first start takes about a minute while the models load.

Options: `run_app.bat --place tokyo_res` picks another place; `--seed N` repeats a layout; `--viewer` also opens the MuJoCo 3D
window. Places: `kazan_novosav` and `tokyo_res` (held out from training in the paper), `akadem`, `akadem_line`,
`berlin_marzahn`, `ekb_center`, `helsinki_kallio`, `kratovo`, `malakhovka`, `malakhovka_line`, `msk_strogino`, `plano_suburb`,
`spb_center`, `spb_krestovsky`, `tlt_industrial`, `warsaw_ursynow`.

On Linux: install PyTorch and `requirements.txt`, then `cd src && OSM2_DIR=../osm2 python demo_server.py --place kazan_novosav`.

## Giving a mission

The page has two modes.

**Plan** has three steps.
1. *Point.* Find the target on the map (hovering over an icon tells what it is, its colour and size; the legend lists all object
   types of this map and highlights them) and click near it. For two steps ("pick up..., then drop...") set two points in order.
2. *Task.* Type the task in Russian or English, or take an example made for the objects of this map. "Parse" shows in plain
   words how the drone understood the text; the parse can be corrected by hand. The page warns if the task has two steps and only
   one point.
3. *Conditions.* A ready set (calm, GPS will be lost, GPS spoofing, storm, fog, weak battery) or a manual choice: wind and
   gusts; fog; GPS loss, spoofing or noise; lidar dirt, rain or interference; a compass bias of 20°; loss of the link to the
   operator; charge at take-off and a cold battery; bird attacks. Then "Fly".

**Fly** shows a large 3D chase view with the map in the corner (the two can be swapped), the drone's state, the core's feelings
grouped by meaning, the event feed, and the drone's camera. "Stop" ends the flight. The simulator runs at about real time, and
the page shows the actual speed. At the end an outcome card says what happened in the world: where the drone landed, where the
load fell, what hangs on the hook, whether there were contacts, and where the drone is now.

## What is inside

- The feeling core learned from consequences: danger in 0.5, 1 and 2 s, "GPS is lying", "lost", "low energy", "strain".
- Somatic markers: before every possible fast action the core estimates its consequences and steers the drone away from a hit.
- Navigation without GPS by lidar and map, search and close check of the target, all six actions, questions to the operator.
- The slow layer uses the explicit energy-to-home telemetry rule, which did better than the learned "low energy" feeling in
  the paper's stage G3.

## Limits

The world is synthetic (MuJoCo) and vision is trained on synthetic frames. Whole-mission tests in the paper included GPS loss
and spoofing, wind and bird attacks; lidar faults, fog, a biased compass and link loss are available in the app but were not part
of those tests, so the drone may do worse with them. Spurious lidar points once made the drone climb without limit; the app now
drops lone lidar points, climbs only on an obstacle seen in three scans in a row, and caps the altitude at 120 m. Free-text
parsing is exact on about half of hand-written tasks, which is why the parse is shown before take-off. Hangar entry rarely
succeeds. Moving to a real drone is described in [REAL_DRONE.md](REAL_DRONE.md).
