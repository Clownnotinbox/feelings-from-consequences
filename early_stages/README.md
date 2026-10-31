# Preliminary study (text only)

Code of the preliminary study in the paper's section "Preliminary study: researcher-set affect gives false ties". A frozen or
LoRA-tuned Qwen2.5 model receives affect signals next to dialogue text, and the scripts test whether the model relies on the
signal or on the words. When the affect labels or the right answers are set by the researcher, words and feelings come out
tied; this is why the drone stages take every target from the world.

- `e*.py`, `r1.py`: one script per experiment (E1–E13, R1); `run_all.py` runs the E2/E3 matrix.
- `audit_*.py`: checks that a signal carries information beyond the words before any training.
- `env*.py`, `bus.py`, `prism_embed.py`, `cremad_prep.py`: environments, the affect bus and dataset preparation.

Public datasets only, each under its own licence: CREMA-D, ESConv and PRISM. Place them under `early_stages/data/`
(`data/prism/conversations.jsonl`, `data/esconv/ESConv.json`, CREMA-D audio for `cremad_prep.py`) and run the scripts from
`early_stages/src/`. The base models (Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct) come from Hugging Face; scripts that
load one take its path with `--model`.
