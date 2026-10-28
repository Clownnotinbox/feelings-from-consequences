"""Run the E2/E3 matrix sequentially (one GPU), skipping finished runs; then summarise."""

import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(a):
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in a.seeds:
        for arm in a.e2_arms:
            jobs.append(("e2", arm, seed))
        for arm in a.e3_arms:
            jobs.append(("e3", arm, seed))
    for exp, arm, seed in jobs:
        out = root / exp / f"{arm}-s{seed}"
        if (out / "RESULT.json").exists():
            continue
        cmd = [
            sys.executable,
            str(HERE / f"{exp}.py"),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--model",
            a.model,
            "--e1",
            a.e1,
            "--replies",
            a.replies,
            "--out",
            str(out),
            "--steps",
            str(a.steps),
            "--opener-weight",
            str(a.opener_weight),
        ]
        if exp == "e2":
            cmd += ["--retention", a.retention]
        if arm in ("BF", "BRF"):
            cmd += ["--format-lora", str(root / "e2" / f"T0-s{seed}" / "lora")]
        if arm in ("DBF", "DWF"):
            cmd += ["--format-lora", str(root / "e3" / f"D0-s{seed}" / "lora")]
        t = time.time()
        print("RUN", exp, arm, seed, flush=True)
        with open(root / f"{exp}-{arm}-s{seed}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        print("DONE", exp, arm, seed, "rc", rc, round(time.time() - t), "s", flush=True)
    subprocess.call([sys.executable, str(HERE / "summarize.py"), str(root)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("model", "e1", "replies", "retention", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--opener-weight", type=float, default=1.0)
    p.add_argument("--e2-arms", nargs="*", default=["T0", "T1", "P", "PF", "B", "BL", "BF", "BRF"])
    p.add_argument("--e3-arms", nargs="*", default=["D0", "DT", "DBL", "DBF", "DWF"])
    main(p.parse_args())
