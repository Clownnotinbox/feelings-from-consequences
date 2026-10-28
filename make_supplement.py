"""Pack the supplementary material of the paper into drone-sim-01/supplement/.
usage: python paper/make_supplement.py"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # drone-sim-01
OUT = ROOT / "supplement"
F08 = ROOT.parent / "affect-foundation-08"


def add_tree(z, folder, arc_root, skip=(".pyc", ".log")):
    # add every file under folder, keep the paths relative to ROOT
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.name.endswith(skip) and "__pycache__" not in p.parts:
            z.write(p, str(Path(arc_root) / p.relative_to(folder)))


def add_files(z, files, arc_root):
    for p in files:
        if p.exists():
            z.write(p, str(Path(arc_root) / p.name))
        else:
            print("missing", p)


# The lab notes were kept in a dialogue with an AI assistant and call the author "the user". The archived copies say "the
# author" instead; dates, hypotheses, thresholds and results stay as written. Each phrase must occur the given number of times.
NEUTRAL = {
    "MISSIONS_DESIGN_RU.md": [
        ("по просьбе пользователя; записано до кода", "по решению автора; записано до кода", 1),
        ("## 0. Задание пользователя", "## 0. Задание автора", 1),
        ("В конце пользователь сам ставит задания", "В конце автор сам ставит задания", 1),
        ("Решения пользователя (26.10)", "Решения автора (26.10)", 1),
        ("пользователь прямо просит довести систему", "автор решил довести систему", 1),
        ("**место для пользователя**", "**место для демонстрации**", 1),
        ("(в демо отвечает пользователь)", "(в демо отвечает оператор)", 1),
        ("демо-окно для пользователя", "демо-окно для оператора", 1),
        ("Пользователь пополнил счёт на 500 ₽ (28.10) и попросил довести", "Автор пополнил счёт на 500 ₽ (28.10) и решил довести", 1),
    ],
    "FEELINGS_DESIGN_RU.md": [
        ("по просьбе пользователя)", "по решению автора)", 1),
        ("Пользователь: «может ядро", "Вопрос автора: «может ядро", 1),
        ("по просьбе пользователя: «где БЯМ", "по вопросу автора: «где БЯМ", 1),
        ("см. ответ пользователю 25.10.", "см. разбор от 25.10.", 1),
        ("по предложению пользователя", "по предложению автора", 1),
        ("Пользователь: цель задаётся", "Замысел автора: цель задаётся", 1),
        ("по решению пользователя", "по решению автора", 1),
        ("без решения пользователя", "без решения автора", 1),
        ("проверяется ваш вопрос:", "проверяется вопрос автора:", 1),
        ("на CPU и вашей 4060.", "на CPU и RTX 4060 ноутбука автора.", 1),
    ],
    "RESULTS_RU.md": [
        ("**По просьбе пользователя: GPS", "**По решению автора: GPS", 1),
        ("(по просьбе пользователя — ядро", "(по решению автора — ядро", 1),
        ("о чём говорил пользователь:", "о чём говорил автор:", 1),
        ("о чём просил пользователь:", "о чём просил автор:", 1),
        ("Задание пользователя (26.10)", "Задание автора (26.10)", 1),
        ("потом пользователь сам ведёт", "потом автор сам ведёт", 1),
        ("Пользователь запустил приложение", "Автор запустил приложение", 1),
    ],
    "REAL_DRONE_RU.md": [("Сценарий пользователя такой.", "Сценарий работы такой.", 1)],
    "NORTH_STAR_RU.md": [("по решению пользователя", "по решению автора", 1)],
    "EXPERIMENT_RULES_RU.md": [
        (
            "Не ввожу пароли и не создаю аккаунты; публикация — силами пользователя, подготовка — моя.",
            "Пароли, аккаунты и публикация — только руками автора; ассистент лишь готовит материалы.",
            1,
        )
    ],
    "FINAL_REPORT_RU.md": [
        ("Гипотеза пользователя", "Гипотеза автора", 1),
        ("пользователя о мухе", "автора о мухе", 1),
    ],
}
DOCS_README = """# Рабочие журналы проекта

Здесь планы каждого этапа (гипотезы и пороги записаны до прогонов, с датами), итоги и правила опытов. Журналы велись по ходу
работы вместе с ИИ-ассистентом (Claude, Anthropic), поэтому местами они написаны от его лица («я» в тексте — ассистент).
В этих копиях автора называют автором, а не «пользователем»; даты, гипотезы, пороги и результаты не менялись.
"""


def add_docs(z, files, arc_root):
    for p in files:
        if not p.exists():
            print("missing", p)
            continue
        s = p.read_text(encoding="utf-8")
        for a, b, n in NEUTRAL.get(p.name, []):
            if s.count(a) != n:
                raise SystemExit(f"{p.name}: {a!r} found {s.count(a)} times, expected {n}")
            s = s.replace(a, b)
        z.writestr(str(Path(arc_root) / p.name), s)
    z.writestr(str(Path(arc_root) / "README_RU.md"), DOCS_README)


OUT.mkdir(exist_ok=True)

# 1. the paper: both PDFs and the LaTeX source
latex = ROOT / "paper" / "latex"
for src, dst in (("main.pdf", "paper_en.pdf"), ("main_ru.pdf", "paper_ru.pdf")):
    if (latex / src).exists():
        shutil.copy(latex / src, OUT / dst)
with zipfile.ZipFile(OUT / "paper_latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(latex.iterdir()):
        if (
            p.suffix in (".tex", ".bib", ".pdf", ".jpg", ".png")
            and not p.name.startswith("main.")
            and not p.name.startswith("main_ru.")
        ):
            z.write(p, p.name)
    for name in ("main.tex", "main_ru.tex"):
        z.write(latex / name, name)

# 2. the code: simulator, mission brain, real-drone bridge, app, analysis, cloud scripts
with zipfile.ZipFile(OUT / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, ROOT / "src", "drone-sim-01/src")
    add_tree(z, ROOT / "real", "drone-sim-01/real")
    add_tree(z, ROOT / "demo", "drone-sim-01/demo")
    add_tree(z, ROOT / "assets", "drone-sim-01/assets")
    for p in sorted((ROOT / "cloud").rglob("*.sh")):
        if ".pem" in p.read_text(encoding="utf-8", errors="ignore"):
            continue  # helper scripts with the path of a private key stay out
        z.write(p, str(Path("drone-sim-01") / p.relative_to(ROOT)))
    add_files(
        z,
        [ROOT / n for n in ("requirements.txt", "install_app.bat", "run_app.bat", "run_demo.bat")]
        + [ROOT / "paper" / "make_figures.py", ROOT / "paper" / "make_supplement.py"],
        "drone-sim-01",
    )
    if (F08 / "src").exists():
        add_tree(z, F08 / "src", "affect-foundation-08/src")

# 3. the data: mission plans, flight results, analysis files, the plans written before each run
with zipfile.ZipFile(OUT / "data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, ROOT / "runs" / "cloud-g", "runs/cloud-g", skip=(".pyc", ".tgz"))
    add_tree(z, ROOT / "runs" / "cloud-f4", "runs/cloud-f4", skip=(".pyc", ".tgz"))
    add_files(z, sorted((ROOT / "runs").glob("*.json")), "runs")
    add_docs(
        z,
        [
            ROOT / n
            for n in (
                "MISSIONS_DESIGN_RU.md",
                "FEELINGS_DESIGN_RU.md",
                "DESIGN_RU.md",
                "RESULTS_RU.md",
                "REAL_DRONE_RU.md",
            )
        ]
        + [
            F08 / n
            for n in ("NORTH_STAR_RU.md", "EXPERIMENT_RULES_RU.md", "PROTOCOL_08_RU.md", "FINAL_REPORT_RU.md")
        ],
        "docs",
    )

# 4. the trained models (the base Qwen2.5-1.5B is downloaded from its own page)
with zipfile.ZipFile(OUT / "models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    add_tree(z, ROOT / "models" / "parser_adapter", "models/parser_adapter")
    add_files(z, [ROOT / "models" / "detector.pt", ROOT / "models" / "attrs_best.pt"], "models")
    add_files(z, [ROOT / "runs" / "core4_s1.runtime.npz", ROOT / "runs" / "markers4_s1.fwd.pkl"], "runs")

for p in sorted(OUT.iterdir()):
    print(f"{p.name:28s} {p.stat().st_size / 1e6:8.1f} MB")
