"""Aggregate RESULT.json files: mean ± half-range over seeds, paired seed bootstrap for key contrasts."""

import json, random, sys
from collections import defaultdict
from pathlib import Path

E2_KEYS = (
    "iid_accuracy",
    "ood_accuracy",
    "free_generation_accuracy",
    "signal_necessity_gain",
    "AR_sensitivity",
    "AR_specificity",
    "AR_correct_after_change",
    "CSE_normalised",
    "irrelevant_lamp_JS",
    "dose_fidelity_mean",
    "conflict_signal_following",
    "shuffle_follows_new_signals",
    "trainable_parameters",
)


def mean(xs):
    return sum(xs) / len(xs)


def boot(diffs, n=5000, seed=0):
    r = random.Random(seed)
    ms = sorted(mean([r.choice(diffs) for _ in diffs]) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n)]]


def main(root):
    root = Path(root)
    lines = ["# Сводка 0.8", ""]
    runs = defaultdict(dict)
    for f in root.glob("e*/*/RESULT.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs[(f.parent.parent.name, r["arm"])][r["seed"]] = r
    e2 = {arm: v for (exp, arm), v in runs.items() if exp == "e2"}
    if e2:
        lines += [
            "## E2 — опора на сигналы (одна реплика)",
            "",
            "|ветвь|" + "|".join(E2_KEYS) + "|",
            "|---" * (len(E2_KEYS) + 1) + "|",
        ]
        for arm in ("T0", "T1", "P", "PF", "B", "BL", "BR", "BF", "BRF"):
            if arm not in e2:
                continue
            cells = []
            for k in E2_KEYS:
                xs = [s[k] for s in e2[arm].values() if k in s]
                cells.append(
                    f"{mean(xs):.3g}" + (f" ±{(max(xs) - min(xs)) / 2:.2g}" if len(xs) > 1 else "")
                    if xs
                    else "-"
                )
            lines.append(f"|{arm} (n={len(e2[arm])})|" + "|".join(cells) + "|")
        lines += ["", "Retention NLL (off / random signals on):"]
        for arm, v in e2.items():
            lines.append(
                f"- {arm}: "
                + ", ".join(
                    f"s{s}: {r['retention_nll']['off']:.3f}/{r['retention_nll']['random_on']:.3f}"
                    for s, r in sorted(v.items())
                )
            )
        lines += ["", "Конфликт по видам:"]
        for arm, v in e2.items():
            lines.append(
                f"- {arm}: "
                + json.dumps(
                    {
                        k: round(mean([r["conflict_by_kind"][k] for r in v.values()]), 3)
                        for k in ("calm_words", "upset_words", "safe_words")
                    }
                )
            )
        for a, b, k in (
            ("B", "T1", "conflict_signal_following"),
            ("B", "T1", "ood_accuracy"),
            ("BL", "T1", "conflict_signal_following"),
            ("BL", "T1", "ood_accuracy"),
            ("B", "BR", "iid_accuracy"),
            ("P", "T1", "ood_accuracy"),
            ("BF", "BRF", "iid_accuracy"),
            ("PF", "P", "ood_accuracy"),
            ("PF", "P", "iid_accuracy"),
            ("BF", "T0", "iid_accuracy"),
            ("BL", "P", "ood_accuracy"),
            ("P", "T1", "iid_accuracy"),
        ):
            if a in e2 and b in e2:
                seeds = sorted(set(e2[a]) & set(e2[b]))
                d = [e2[a][s][k] - e2[b][s][k] for s in seeds]
                if d:
                    lines.append(
                        f'- {a}−{b} {k}: {mean(d):+.3f} (seeds {seeds}, bootstrap {boot(d) if len(d) > 1 else "n/a"})'
                    )
    e3 = {arm: v for (exp, arm), v in runs.items() if exp == "e3"}
    if e3:
        lines += [
            "",
            "## E3 — дофамин как единственный источник",
            "",
            "|ветвь|accuracy|после смены +0/+1/+2|erase accuracy|teacher|params|",
            "|---|---|---|---|---|---|",
        ]
        for arm in ("D0", "DT", "DB", "DBL", "DW", "DBF", "DWF", "DP", "DWP"):
            if arm not in e3:
                continue
            v = list(e3[arm].values())
            cl = [r["closed_loop"] for r in v]
            er = [r["closed_loop_dopamine_erased"] for r in v]
            since = " / ".join(
                f"{mean([c['accuracy_by_turns_since_flip'].get(str(k), 0) for c in cl]):.2f}"
                for k in (0, 1, 2)
            )
            lines.append(
                f"|{arm} (n={len(v)})|{mean([c['accuracy'] for c in cl]):.3f}|{since}|{mean([c['accuracy'] for c in er]):.3f}|"
                f"{mean([c['teacher_policy_accuracy_on_same_trajectories'] for c in cl]):.3f}|{v[0]['trainable_parameters']}|"
            )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
