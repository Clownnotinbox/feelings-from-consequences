# Hypotheses and rules written before the runs

Every stage was planned in writing before its runs: the hypotheses, the measures, the thresholds and the stopping rules, with
the date and time. Thresholds were never changed after the results. This file lists them in English with their outcomes. The
original dated plans (in Russian) are in `docs/` inside `data.zip` of
[release v1.0](https://github.com/Clownnotinbox/feelings-from-consequences/releases/tag/v1.0): `FEELINGS_DESIGN_RU.md` for
stages A–E ("plan A–E" in code comments) and `MISSIONS_DESIGN_RU.md` for stages F–G ("plan F–G"); section numbers below refer
to these files. Runs that were not pre-registered are marked *post hoc*. All dates are in 2026.

## Stages A–C: the feeling core and somatic markers (plan A–E §4, 23.10)

- H4: the core raises the alarm earlier than a classic time-to-collision threshold.
- H5: a route memory raises the share of arrivals without GPS.
- H1–H3 concerned a language model reading feelings under a token budget; they were superseded by stage D.
- Measures: arrivals, contacts (counted as events), error in metres, warning margin, calibration. Test worlds are seeds never
  used in training; three training seeds; at least 200 flights per condition; bootstrap confidence intervals.

Outcome (paper, Section "Fast feelings"): somatic markers cut hits on the static world in closed-loop flight by 92 %
(familiar places) and 85 % (unseen places), mean of three seeds.

## Stage D: slow decisions with a language model (plan A–E §12.4–12.5, 24.10)

- H8: a core that calls the LLM only when needed uses far fewer calls at the same safety; on the slow level, a small policy on the
  core's feelings decides alone and hands the least certain share of decisions to the LLM.
- H9 (main): an LLM given the core's feelings chooses better than the same LLM given raw telemetry from the same sources, at an
  equal token budget; both against no sensors.
- H10 (lesion): removing one feeling hurts exactly the decisions that depend on it.
- H11 (trust): after fine-tuning on world outcomes, the LLM relies on a feeling only as far as it is right.
- Feelings are given in words and as soft embedding slots; the control is the same slots with raw telemetry.

Outcome: H9 not confirmed (feelings minus telemetry ≈ 0). The core alone beats every LoRA LLM. Feelings as soft slots beat the
same feelings in words.

## Stage D2: messages about future events (plan A–E §12.5, H12)

- H12: an LLM is needed where knowledge is not in the sensors: a message in words about the future of the world; the core plus
  the LLM should beat each alone.

Outcome: H12 not confirmed; the LLMs did not use the messages, but trusted the core's per-option judgement.

## Stage D3: strong events, novel wording, a gate (plan A–E §12.6, 25.10)

- Pilot rule: if the announced event changes the best plan in fewer than 20 % of decisions, words have nothing to carry, and
  no data or GPU are spent.
- H13 (LLM helps the core), H14 (understanding, not keywords), H15 (double dissociation under lesions), H16 (a gate calls the LLM
  rarely and keeps ≥ 90 % of the gain).

Outcome: the pilot rule stopped the stage (the event changed the best plan in 15.2 % of decisions); H13–H16 were not run.

## Stage E: the goal given by description (plan A–E §13.1, 13.3, 25.10)

- H17: on visible scenes with look-alikes, a vision-language model with a frame finds the target more often than coordinates
  alone. H18: with novel wording, it beats keywords with perfect vision. H19: its accuracy grows with the target's size.
- Go rule to the closed loop: VLM with novel wording ≥ 0.6 and better than the best rival without an LLM by ≥ 0.15, lower 95 %
  bound > 0.
- Zoom check (§13.3): H20, close-ups beat marks without them; H21, close-ups beat coordinates alone by ≥ 0.15; same go rule.

Outcome: H17 and H18 rejected, H19 confirmed in direction; the go rule failed (0.19). With close-ups: H20 not confirmed, H21 not
confirmed by its threshold (+0.12); the go rule failed (0.50). No closed loop with a VLM was built.

## Stage F: missions in real places (plan F–G §6, 26.10; amendments §6.1–6.5)

| check | rule | outcome |
|---|---|---|
| F1 perception | recall of the object type ≥ 0.9 for objects ≥ 30 px, precision ≥ 0.9, colour ≥ 0.9 | passed (0.974, 0.978, 0.958) |
| F2 target choice | hard scenes with look-alikes, novel wording: ≥ 0.85 and better than coordinates alone by ≥ 0.3 | failed (0.55; 0.72 with the true parse) |
| F3 parser | all fields correct ≥ 0.9 on novel wording and ≥ 0.85 on 100 hand-written tasks | failed (0.93 familiar, 0.17 novel, 0.21 hand-written) |
| F3b parser on paraphrases (§6.2, 26.10 21:50) | same thresholds | failed (0.56 hand-written, 0.43 novel) |
| F4 missions (§6, §6.1) | wrong target ≤ 2 % | failed (3.2 %) |
| F4b (§6.3, 27.10) | the fixed system: touch only targets the description singles out | *post hoc*: wrong target 1.6 % |
| F5 lesions (§6, §6.3) | (c) without the core, more contacts; (b) without feelings, fewer successes without GPS and at low charge | (c) confirmed: static-contact missions 5.4 % → 11.0 %; (b) not confirmed |
| F4c (§6.4) | the system with an ideal operator | *post hoc*: steps 0.545 vs 0.299, 0 wrong targets |
| F4d (§6.5, 27.10 20:10) | the fixed system alone on 300 F4 missions; wrong target ≤ 2 % | *post hoc*: 0.67 % |

## Stage G: final test (plan F–G §6.6, 28.10)

| part | rule | outcome |
|---|---|---|
| G1 hangar door finder | the new finder stays only if it enters no less often and finds no door on a closed hangar | the new finder found doors on 3 of 14 closed hangars; the old one stays |
| G2 final test, 300 fresh missions | wrong target alone ≤ 2 % at the upper 95 % bound; replication: fewer static-contact missions with the core than with the reflex, interval excluding 0 | replication confirmed (4.3 % vs 14.7 %); the wrong-target rule failed (1.0 %, upper bound 2.3 %) |
| G3 slow feelings in hard missions | (1) without a slow layer, more batteries dead in the air than with the full system; (2) H6: feelings no worse than telemetry; the plan recorded the expectation that H6 would fail on energy | (1) not confirmed (29 % vs 29 %); (2) H6 not confirmed: the telemetry rule 16 % vs 29 % |
| G4 false and delayed danger | false alarms: more overrides, fewer steps, contacts not fewer; delay of 0.3 s: more static-contact missions | false alarms: +24 overrides per mission, +5 pp static-contact missions, no step cost; delay: +3 pp, not significant |
| G5 PX4 readiness check | not a hypothesis | the link works; 1 of 12 missions on PX4 vs 7 of 12 on our controller |

Notes recorded before the results: the first G launch ran out of GPU memory and its rows were discarded whole; G3 and G4 were
shortened for budget (G3 to 55 min, G4 to the first 100 missions), with measures and thresholds unchanged.
