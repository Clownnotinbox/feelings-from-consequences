"""One request -> a real flight.
usage: run_real_mission.py --text "..." --point LAT,LON [--point LAT,LON] [--area LAT,LON] [--mavlink udpin:0.0.0.0:14540]
       [--autopilot px4|ardupilot] [--parser ADAPTER --base BASE | --parsed parse.json] [--payload-kg 0.5] [--dry]
"""

import argparse, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))


def parse_text(text, adapter, base):
    import mission_parser as MP, train_parser as TP, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    lm = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu"
        ),
        adapter,
    ).eval()
    p = MP.parse_output(TP.generate(lm, tok, [{"text": text}], bs=1)[0])
    return p if MP.well_formed(p) else None


def confirm(parsed):
    print("\n=== what I understood ===\n" + json.dumps(parsed, ensure_ascii=False, indent=1))
    a = input("Fly this? [y = yes / e = edit the JSON in parse_edit.json / n = no]: ").strip().lower()
    if a == "e":
        Path("parse_edit.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=1), encoding="utf-8")
        input("edit parse_edit.json, then press Enter... ")
        return json.loads(Path("parse_edit.json").read_text(encoding="utf-8"))
    return parsed if a == "y" else None


def operator_cli(env):
    """The drone's question."""
    from PIL import Image

    Path("asks").mkdir(exist_ok=True)

    def ask(step, cands):
        print(f"\n=== the drone asks: which one is '{step['target'].get('kind')}' {step['target']} ? ===")
        for k, (p, h, _) in enumerate(cands):
            print(f'  {k + 1}: {h["cls"]} at ({h["x"]:.1f}, {h["y"]:.1f}), score {p:.2f}')
            yaw = math.degrees(
                math.atan2(h["y"] - env.data.qpos[1], h["x"] - env.data.qpos[0]) - env.obs_yaw()
            )
            env.set_gimbal(yaw, -35.0)
            rgb, _, _ = env.gimbal_frame(depth=False)
            Image.fromarray(rgb).save(f"asks/cand_{int(time.time())}_{k + 1}.jpg")
        a = input("number (0 = none of these, Enter = no answer): ").strip()
        return int(a) - 1 if a.isdigit() else None

    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--point", action="append", required=True)
    ap.add_argument("--area")
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14540")
    ap.add_argument("--autopilot", default="px4")
    ap.add_argument("--parser")
    ap.add_argument("--base")
    ap.add_argument("--parsed")
    ap.add_argument("--payload-kg", type=float, default=0.5)
    ap.add_argument("--name", default="site")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pts = [tuple(map(float, p.split(","))) for p in a.point]
    lat0, lon0 = map(float, a.area.split(",")) if a.area else pts[0]
    import osm_fetch2, city3d, real_env as RE

    osm_fetch2.AREAS[a.name] = (
        lat0,
        lon0,
        "real",
        "operator_area",
    )  # the operator's area: fetched once (Overpass), then cached
    if not (city3d.OSM2 / f"{a.name}.json").exists():
        city3d.OSM2.mkdir(parents=True, exist_ok=True)
        osm_fetch2.fetch(a.name, lat0, lon0, city3d.OSM2)
    city3d.AREAS[a.name] = osm_fetch2.AREAS[a.name]
    W, _, _ = city3d.build(a.name, 0, map_errors=False, actors=False)
    MAP = W.map
    frame = RE.LocalFrame(lat0, lon0)
    parsed = (
        json.loads(Path(a.parsed).read_text(encoding="utf-8"))
        if a.parsed
        else parse_text(a.text, a.parser, a.base)
    )
    parsed = confirm(parsed) if parsed else None
    if parsed is None:
        print("not flying: no confirmed mission")
        return
    rough = [frame.xy(la, lo) for la, lo in pts]
    print("rough points in the map frame:", [(round(x, 1), round(y, 1)) for x, y in rough])
    if a.dry:
        print("dry run: map, parse and points are ready; not connecting to the drone")
        return
    import mav as MV, mission_brain as MB, nav as NV, safety as SF, perceive as PC

    link = MV.Link(a.mavlink, a.autopilot)
    env = RE.RealEnv(link, RE.Ros2Livox(), RE.RtspCamera(), frame, payload_kg=a.payload_kg)
    o = env.reset()
    models = HERE.parent / "models"
    perc = PC.Perceiver(str(models / "detector.pt"), str(models / "attrs_best.pt"))
    saf = SF.Safety(env)
    nav = NV.Navigator(MAP, env.home, seed=0)
    brain = MB.Brain(
        env, MAP, parsed, rough, perc, safety=saf, nav=nav, log=print, ask=operator_cli(env), robust=True,
        slow="telemetry",  # stage G3: an explicit energy-to-home rule beat the learned slow feelings
    )
    try:
        while not brain.done:
            o = env.step(brain.act(o))
    except KeyboardInterrupt:
        print("operator stop: the autopilot returns to launch")
    finally:
        env.finish()
        env.close()
    print("skipped steps:", brain.skipped, "notes:", *[n[1] for n in brain.notes[-20:]], sep="\n")


if __name__ == "__main__":
    main()
