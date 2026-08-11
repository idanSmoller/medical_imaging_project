# Overleaf report

`main.tex` is a self-contained skeleton of the final report. Upload just that
one file to a new Overleaf project (or paste it into a blank one) — it compiles
with pdfLaTeX, no bibtex step and no extra packages to install.

## How it maps to the course instructions

The sections are exactly the six the instructions require, in order: title page,
introduction, proposed extension, methodology, results and analysis, conclusion.
Every sub-bullet in the instructions has a matching bullet here. Hard limit is
**6 pages**, so plan to cut, not to add.

## How to use it

Each bullet is a *thing to say*, not final text. Turn bullets into prose as the
work lands; delete any bullet that turns out not to be worth a sentence.

Bullets marked `[TODO]` need numbers that do not exist yet — they are the
results of the three ablation runs (models A / B / C). Everything else can be
written now, before training finishes.

## What is not yet true

The report claims three trained models and a metrics table. As of now the repo
has the disagreement head and losses implemented but never trained end-to-end,
and no generalized energy distance. See "Current state" in `../AGENTS.md`.
Write sections 1–4 now; sections 5–6 need the runs.
