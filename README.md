# Feelings Learned from Consequences

Code for the paper *Feelings Learned from Consequences: A Somatic-Marker Core for Language-Instructed Drone Missions, and Where
Language Models Do Not Help* (Mark Kolpakov, 2026). The paper source is in [`paper/latex`](paper/latex), the built PDFs are
`paper/latex/main.pdf` (English) and `paper/latex/main_ru.pdf` (Russian).

A drone in a MuJoCo simulator of real OpenStreetMap districts learns "feelings" (danger of a contact within 2 s for each of seven
candidate actions, GPS distrust, being lost, low energy) only from what the world does next: an exact copy of the simulator is
forked and every candidate action is played forward. A mission brain flies language-instructed missions (land, drop by parachute,
pick up, enter a hangar, inspect, fly by) and uses the feeling core as its fast safety layer. Lesions replace the core with a
classic reflex, or replace the feelings with raw telemetry. Language models are tested as the slow decision-maker on top.

## Layout

| path | what |
|---|---|
| `src/` | simulator (`mission_env.py`, `drone_env.py`, `osm_world.py`), feeling core and consequence model (`stage_a.py`, `core_runtime.py`, `markers_geom.py`), mission brain (`mission_brain.py`, `safety.py`, `nav.py`), perception, parser, evaluation and analysis scripts (`eval_missions.py`, `analyze_*.py`), the app server (`demo_server.py`) and its browser test (`app_test.py`) |
| `demo/` | the app page |
| `real/` | MAVLink bridge for PX4/ArduPilot and the PX4 SITL co-simulation |
| `cloud/` | the scripts each stage ran on the cloud GPU machines |
| `osm2/` | the map data of all places (© OpenStreetMap contributors, ODbL) |
| `assets/` | textures |
| `paper/` | LaTeX source, figures, figure scripts |
| `docs/` | the plan of every stage with its hypotheses and thresholds written before the runs, the results log, the real-drone readiness notes, the video list (in Russian, `VIDEOS_EN.md` in English) |
| `early_stages/` | code of the preliminary text-only study |

Flight logs, mission plans and analysis outputs (`data.zip`), the trained models (`models.zip`) and the videos are archived
separately; see the paper's section "Code, data and videos".

## Running the app (Windows, NVIDIA GPU, Python 3.11)

1. Unpack `models.zip` into this folder, so that `models/` and `runs/` sit next to `src/`.
2. Run `install_app.bat` once. It creates `.venv`, installs the packages and downloads Qwen2.5-1.5B-Instruct for the parser.
3. Run `run_app.bat` and open `http://localhost:8765`. Options: `--place tokyo_res`, `--seed N`, `--viewer`. The page has an
   RU/EN switch in its header.

On Linux: `pip install -r requirements.txt` plus PyTorch, then `cd src && OSM2_DIR=../osm2 python demo_server.py --place kazan_novosav`.

## Reproducing the numbers

The final test is `src/eval_missions.py` driven by `cloud/stageg/` (300 missions per arm); `src/analyze_g.py` turns the logs into
`G_ANALYSIS.json`, from which every stage G number in the paper is copied. Replays on a different GPU model differ in detail
because of numerical differences in the detector.

## Licences

Code: AGPL-3.0 (it calls Ultralytics YOLO11, AGPL-3.0). Models and data: CC BY 4.0, except the detector (AGPL-3.0). Map data
© OpenStreetMap contributors, ODbL. MuJoCo: Apache-2.0. Qwen2.5 models: Apache-2.0.

## Citation

See `CITATION.cff`.
