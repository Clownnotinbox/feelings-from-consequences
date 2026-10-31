# Feelings Learned from Consequences

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23740970.svg)](https://doi.org/10.5281/zenodo.23740970)

Code for the paper *Feelings Learned from Consequences: A Somatic-Marker Core for Language-Instructed Drone Missions, and Where
Language Models Do Not Help* (Mark Kolpakov, 2026). The paper is in [`paper/latex`](paper/latex): `main.pdf` in English and
`main_ru.pdf`, the author's Russian version.

A drone in a MuJoCo simulator of real OpenStreetMap districts learns "feelings" (danger of a contact within 2 s for each of seven
candidate actions, GPS distrust, being lost, low energy) only from what the world does next: an exact copy of the simulator is
forked and every candidate action is played forward. A mission brain flies language-instructed missions (land, drop by parachute,
pick up, enter a hangar, inspect, fly by) and uses the feeling core as its fast safety layer. Lesions replace the core with a
classic reflex, or replace the feelings with raw telemetry. Language models are tested as the slow decision-maker on top.

## Layout

| path | contents |
|---|---|
| `src/` | simulator (`mission_env.py`, `drone_env.py`, `osm_world.py`, `city3d.py`), feeling core and consequence model (`stage_a.py`, `train_core.py`, `train_markers.py`, `core_runtime.py`, `markers_geom.py`), mission brain (`mission_brain.py`, `safety.py`, `nav.py`), perception, parser, target choice, evaluation and analysis (`eval_missions.py`, `analyze_*.py`), the app server (`demo_server.py`) and its browser test (`app_test.py`), video composition |
| `demo/` | the app page |
| `real/` | MAVLink bridge for PX4/ArduPilot, the real-drone runner and the PX4 SITL co-simulation |
| `scripts/cloud/` | the scripts each stage ran on single-GPU cloud machines, one folder per stage |
| `osm/`, `osm2/` | map data of the places (© OpenStreetMap contributors, ODbL) |
| `assets/` | textures |
| `paper/` | LaTeX source of both versions, figures, `make_figures.py` |
| `docs/` | [hypotheses and rules written before the runs](docs/PREREGISTRATION.md), [the app](docs/APP.md), [moving to a real drone](docs/REAL_DRONE.md), [the videos](docs/VIDEOS.md) |
| `early_stages/` | the preliminary text-only study |

Flight logs, mission plans and analysis outputs (`data.zip`), the trained models (`models.zip`) and the videos (`videos.zip`)
are attached to [release v1.0](https://github.com/Clownnotinbox/feelings-from-consequences/releases/tag/v1.0). `data.zip` also
holds the original dated plans of every stage.

## Running the app

Windows with an NVIDIA GPU and Python 3.11:

1. Unpack `models.zip` from the release into the repository root, so that `models/` and `runs/` sit next to `src/`.
2. Run `install_app.bat` once. It creates `.venv`, installs the packages and downloads Qwen2.5-1.5B-Instruct for the parser.
3. Run `run_app.bat` and open `http://localhost:8765`. Options: `--place tokyo_res`, `--seed N`, `--viewer`.

Linux: install PyTorch and `requirements.txt`, download Qwen2.5-1.5B-Instruct into `models/qwen1.5b`, then
`cd src && python demo_server.py --place kazan_novosav`.
Details in [docs/APP.md](docs/APP.md).

## Reproducing the numbers

Every stage ran on a cloud machine with the repository at `~/af08`; the stage's script in `scripts/cloud/` installs the
environment and runs it. The final test is `scripts/cloud/g_final/g.sh` (300 missions per arm through `src/eval_missions.py`);
`src/analyze_g.py` turns the logs into `G_ANALYSIS.json`, from which every stage G number in the paper is copied. The flight logs
and analysis files are in `data.zip`. Replays on a different GPU model differ in detail because of numerical differences in the
detector.

## Licences

Code: AGPL-3.0 (it calls Ultralytics YOLO11, AGPL-3.0). Models and data: CC BY 4.0, except the detector (AGPL-3.0). Map data
© OpenStreetMap contributors, ODbL. Fonts in `src/fonts` (Inter, IBM Plex Mono): SIL Open Font License 1.1. MuJoCo:
Apache-2.0. Qwen2.5 models: Apache-2.0.

## Citation

See `CITATION.cff`.
