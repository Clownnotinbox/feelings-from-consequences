"""Figures of the paper, in English and in Russian.
usage: python make_figures.py   -> paper/fig_*.png, latex/fig_*.pdf (Russian ones end with _ru)"""

import json
from pathlib import Path
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)

RU = False  # the language of the figure being drawn


def tr(en, ru):
    # pick the text for the current language
    return ru if RU else en


def num(v, fmt="{:.2f}"):
    # Russian papers write a decimal comma
    s = fmt.format(v)
    return s.replace(".", ",") if RU else s


def save(fig, name):
    end = "_ru" if RU else ""
    fig.savefig(HERE / (name + end + ".png"))
    fig.savefig(HERE / "latex" / (name + end + ".pdf"))
    plt.close(fig)


def comma_ticks(ax):
    # decimal commas on the x axis too
    if RU:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))


def bars(ax, labels, vals, colors, err=None, fmt="{:.2f}"):
    y = range(len(labels))
    ax.barh(
        list(y),
        vals,
        color=colors,
        height=0.62,
        xerr=err,
        error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3},
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(
            v + max(vals) * 0.02 + (err[1][i] if err is not None else 0),
            i,
            num(v, fmt),
            va="center",
            color=INK,
            fontsize=9,
        )
    comma_ticks(ax)


def fig_closed_loop():
    """Static-world contacts per 40-s flight."""
    lab = [
        tr("pilot only", "только пилот"),
        tr("classic reflex", "обычный рефлекс"),
        tr("GRU markers", "маркеры GRU"),
        tr("consequence model\n(3 seeds)", "модель последствий\n(3 сида)"),
    ]
    fam = [1.32, 0.76, 0.78, (0.07 + 0.10 + 0.15) / 3]
    uns = [1.27, 0.99, 0.77, (0.21 + 0.19 + 0.18) / 3]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.8), sharey=True)
    for ax, v, t in (
        (axs[0], fam, tr("familiar places", "знакомые места")),
        (axs[1], uns, tr("unseen places", "незнакомые места")),
    ):
        bars(ax, lab, v, [GREY, ORANGE, "#c3c2b7", BLUE])
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(tr("static-world contacts per flight", "касаний неподвижного за полёт"))
        ax.set_xlim(0, 1.6)
    fig.suptitle(
        tr(
            "Closed loop, 120 missions per split: the core nearly removes hits on the static world",
            "Замкнутый контур, 120 миссий на часть: ядро почти убирает удары о неподвижное",
        ),
        color=INK,
        fontsize=10.5,
        x=0.02,
        y=1.04,
        ha="left",
    )
    save(fig, "fig_closed_loop")


def fig_llm():
    """Stage D v2: share of best choices (mean of 3 seeds), results log 25.10 00:40."""
    lab = [
        tr("LLM: mission only", "БЯМ: только миссия"),
        tr("LLM: feelings as words", "БЯМ: чувства словами"),
        tr("LLM: feelings as slots", "БЯМ: чувства слотами"),
        tr("LLM: raw telemetry", "БЯМ: сырая телеметрия"),
        tr("LLM: telemetry + feelings", "БЯМ: телеметрия и чувства"),
        tr("core alone", "ядро само"),
    ]
    fam = [0.50, 0.51, 0.53, 0.63, 0.65, 0.69]
    uns = [0.47, 0.44, 0.52, 0.57, 0.52, 0.66]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.9), sharey=True)
    for ax, v, t in (
        (axs[0], fam, tr("familiar places", "знакомые места")),
        (axs[1], uns, tr("unseen places", "незнакомые места")),
    ):
        bars(ax, lab, v, [GREY] * 5 + [BLUE])
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(tr("share of best choices", "доля лучших выборов"))
        ax.set_xlim(0, 0.8)
    fig.suptitle(
        tr(
            "Slow decisions: the core decides at least as well as any fine-tuned LLM",
            "Медленные решения: ядро решает не хуже любой дообученной БЯМ",
        ),
        color=INK,
        fontsize=10.5,
        x=0.02,
        y=1.04,
        ha="left",
    )
    save(fig, "fig_llm_decisions")


def fig_missions():
    """Whole missions: missions with a static contact, with vs without the core."""
    F = json.load(open(RUNS / "cloud-f4" / "F4B_ANALYSIS.json", encoding="utf-8"))
    rows = [
        (
            tr("F5, 500 missions", "F5, 500 миссий"),
            F["full"]["missions_with_static_hit"],
            F["reflex"]["missions_with_static_hit"],
        )
    ]
    gp = RUNS / "cloud-g" / "G_ANALYSIS.json"
    if gp.exists():
        G = json.load(open(gp, encoding="utf-8"))
        if "G_full" in G and "G_reflex" in G:
            n = G["G_full"]["missions"]
            rows.append(
                (
                    tr(f"G2, {n} new missions", f"G2, {n} новых миссий"),
                    G["G_full"]["missions_with_static_hit"],
                    G["G_reflex"]["missions_with_static_hit"],
                )
            )
    fig, ax = plt.subplots(figsize=(6.4, 1.2 + 0.75 * len(rows)))
    for i, (name, a, b) in enumerate(rows):
        for j, (v, c, l) in enumerate(
            (
                (a, BLUE, tr("with the feeling core", "с ядром чувств")),
                (b, ORANGE, tr("without the core (classic reflex)", "без ядра (обычный рефлекс)")),
            )
        ):
            y = i * 2.6 + j
            ax.barh(
                y,
                v[0] * 100,
                color=c,
                height=0.8,
                label=l if i == 0 else None,
                xerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]],
                error_kw={"ecolor": INK2, "capsize": 3},
            )
            ax.text(v[2] * 100 + 0.4, y, num(v[0] * 100, "{:.1f}") + " %", va="center", color=INK, fontsize=9)
    ax.set_yticks([i * 2.6 + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(
        tr(
            "missions with a contact with the static world, % [95 % CI]",
            "миссий с касанием неподвижного, % [95 % ДИ]",
        )
    )
    ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2 - 0.03 * len(rows)), ncol=2, fontsize=8.5
    )
    ax.set_xlim(0, max(r[2][2] for r in rows) * 100 + 4)
    ax.set_title(
        tr(
            "Whole missions in unseen real places: without the core, static contacts are 2 to 3 times as frequent",
            "Целые миссии в незнакомых местах: без ядра касаний неподвижного в 2–3 раза больше",
        ),
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    save(fig, "fig_missions_lesion")


def fig_actions():
    """Step success by action: the drone alone vs with a person answering its questions (G2)."""
    gp = RUNS / "cloud-g" / "G_ANALYSIS.json"
    if not gp.exists():
        return
    G = json.load(open(gp, encoding="utf-8"))
    if "G_full" not in G:
        return
    acts = ["flyby", "inspect", "land", "drop", "pickup", "enter"]
    names = {
        "flyby": tr("fly by", "пролёт"),
        "inspect": tr("inspect (orbit)", "осмотр (облёт)"),
        "land": tr("land", "посадка"),
        "drop": tr("parachute drop", "сброс на парашюте"),
        "pickup": tr("hook pick-up", "захват крюком"),
        "enter": tr("enter hangar", "влёт в ангар"),
    }
    alone = G.get("G_full_same_as_operator", G["G_full"])  # the same missions as the operator run
    a = [alone["by_action"].get(k, [0, 1, 0])[2] for k in acts]
    b = [G.get("G_operator", {}).get("by_action", {}).get(k, [0, 1, None])[2] for k in acts]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = range(len(acts))
    ax.barh(
        [i - 0.2 for i in y],
        a,
        height=0.38,
        color=BLUE,
        label=tr(
            "alone (one request, no questions answered)", "сам (один запрос, на вопросы никто не отвечает)"
        ),
    )
    if any(v is not None for v in b):
        ax.barh(
            [i + 0.2 for i in y],
            [v or 0 for v in b],
            height=0.38,
            color=AQUA,
            label=tr("with a person answering its questions", "с человеком, который отвечает на вопросы"),
        )
    for i in y:
        ax.text(a[i] + 0.01, i - 0.2, num(a[i]), va="center", fontsize=8.5, color=INK)
        if b[i] is not None:
            ax.text(b[i] + 0.01, i + 0.2, num(b[i]), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([names[k] for k in acts])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    comma_ticks(ax)
    ax.set_xlabel(tr("share of steps done (judged by the world)", "доля выполненных шагов (решает мир)"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=1)
    ax.set_title(
        tr(
            f"Final test, the same {alone['missions']} fresh missions in unseen places",
            f"Итоговая проверка, те же {alone['missions']} новых миссий в незнакомых местах",
        ),
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    save(fig, "fig_actions")


def fig_slow():
    """G3: hard missions, the slow feelings switched off one by one."""
    gp = RUNS / "cloud-g" / "G_ANALYSIS.json"
    if not gp.exists():
        return
    G = json.load(open(gp, encoding="utf-8"))
    if "G3_full" not in G:
        return
    keys = ["full", "telemetry", "slow_none", "nolost", "noenergy"]
    names = {
        "full": tr("feelings of the core", "чувства ядра"),
        "telemetry": tr("raw telemetry instead", "сырая телеметрия вместо них"),
        "slow_none": tr("no slow layer", "без медленного слоя"),
        "nolost": tr('without "I am lost"', "без «я потерялся»"),
        "noenergy": tr('without "low energy"', "без «мало сил»"),
    }
    colors = [BLUE, ORANGE, GREY, "#c3c2b7", "#c3c2b7"]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 2.6), sharey=True)
    panels = (
        ("dead_in_air", tr("battery dead in the air", "батарея села в воздухе")),
        ("home_landed", tr("came back and landed at home", "вернулся и сел дома")),
    )
    for ax, (m, t) in zip(axs, panels):
        v = [G[f"G3_{k}"][m][0] for k in keys]
        lo = [G[f"G3_{k}"][m][0] - G[f"G3_{k}"][m][1] for k in keys]
        hi = [G[f"G3_{k}"][m][2] - G[f"G3_{k}"][m][0] for k in keys]
        bars(ax, [names[k] for k in keys], v, colors, err=[lo, hi])
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(tr("share of missions [95 % CI]", "доля миссий [95 % ДИ]"))
        ax.set_xlim(0, 1)
    n = G["G3_full"]["missions"]
    fig.suptitle(
        tr(
            f"Hard missions ({n}): low battery, cold, GPS lost or spoofed, wind, return home always asked",
            f"Тяжёлые миссии ({n}): мало заряда, холод, отказ или подмена GPS, ветер, возврат обязателен",
        ),
        color=INK,
        fontsize=10.5,
        x=0.02,
        y=1.06,
        ha="left",
    )
    save(fig, "fig_slow")


if __name__ == "__main__":
    for RU in (False, True):
        fig_closed_loop()
        fig_llm()
        fig_missions()
        fig_actions()
        fig_slow()
    print("figures:", sorted(p.name for p in HERE.glob("fig_*.png")))
