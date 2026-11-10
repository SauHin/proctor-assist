# CLAUDE.md: Cheating Detection for Online Exam Proctors

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). Owner: Jonathan Davin.
`docs/PLAN.md` holds the project's goals, scope, and constraints. Read it before starting any task.

## What this project is

A system that watches the webcam video of online exam takers and flags moments for a
proctor to review. The proctor decides. The system never labels a person as a cheater.

Single experiment track (PLAN §5):
- R0: head-turn rule without training (reference).
- B1: small CNN per frame, trained from scratch (DL baseline).
- P1: MobileNetV2 per frame, set up as in the reference paper (proven method and app
  fallback).
- E1: frame features + GRU over a window.
- E2: landmarks + GRU.
- E3: pretrained video model (optional).

Every model gives one suspicion score from 0 to 1 per window (`docs/decisions.md` D10).
Main metric: cheating events detected vs false alarms per hour. Also CPU scalability and
tests under changed conditions.

Data: MSU Online Exam Proctoring (OEP) dataset, webcam video only (Kaggle mirror
`raajanwankhade/oep-dataset`). Ground truth = the OEP cheating intervals, with an
owner-decided mapping from OEP types to project classes.

The package is `src/proctor_assist/`. Why this topic was chosen: `docs/decisions.md` D9.

`docs/PLAN.md` fixes goals, scope, and constraints. Implementation details are open. Decide
or discuss them as "How to work with me" says, and record each decision in
`docs/decisions.md`.

## Scope rules

- Out of scope:
  - Audio and the wearable camera of OEP.
  - Face recognition or identity verification.
  - Screen or browser monitoring, and offline exams in a classroom.
  - LLM API features, and storage of video by the app.
- If a task seems to need something outside the plan, raise it instead of adding it.

## Integrity rules (non-negotiable)

This project is graded, and fabricated evidence is an academic integrity violation.

- Never write, estimate, or "fill in" any metric, training time, or result. Every number in
  docs, READMEs, figures, or the report must come from a saved result file produced by
  running code. If a result does not exist yet, write `TBD` and say so.
- Never decide the final class mapping (OEP types to project classes), ground-truth labels,
  or any tester or user responses. The owner decides each point in the chat, from
  multiple-choice questions. After that, you can record the answers of the owner in
  `docs/decisions.md`, word for word. Mark them as "decided by the owner in chat" with the
  date. You can also write the files that follow from these answers. Never pick an option yourself,
  and never record a check that the owner did not report. A draft stays in a separate file
  that is marked `PROPOSAL`. (Changed by the owner on 2026-11-10, D16.)
- Never generate fake or synthetic data and present it as real.
- After substantial help, add a short entry to `docs/ai_usage_log.md` yourself, and follow
  the rules section in that file. Do not ask the owner for missing details. For an unknown
  entry number, write `TBD`. In Student Verification, write `PENDING: <plan>`. Never fill
  in verification, test results, or dates that you do not know. The owner reviews all
  entries at the end.

## Privacy rules

- The app processes video in memory and never stores it.
- Never commit OEP videos, frames, or landmark files. The repository is public (D1).
- Webcam tests use only team members who consent, and nothing is recorded.

## Data leakage rules

- Subjects are split (fixed seed) **before** any frame, window, or landmark sequence is
  extracted. Training windows come only from training subjects.
- Normalization statistics, thresholds, and any fitted statistic are fit on training or
  validation subjects, never on test subjects.
- Alarm thresholds are chosen on validation subjects only (see PLAN §6).
- Keep a test for this rule. It fails on any subject in more than one split.

## Compute and workflow (local first, Colab for GPU only; `docs/decisions.md` D8)

- Work that needs no GPU runs locally in `.venv`: data prep, EDA, CPU evaluation, and all
  tests. Without `PROCTOR_ROOT`, data and run folders live in the repo's gitignored `data/` and
  `runs/`.
- Only GPU jobs (model training, and encoding that is too slow on CPU) run in Google Colab Pro
  through the Colab MCP server (`googlecolab/colab-mcp`), connected to a Colab notebook open in
  the browser. Run `pytest` in Colab before a GPU job.
- The git repo is the single source of truth. Never write core logic directly into notebook
  cells. Flow for a GPU job: edit locally → commit and push → in Colab: `git pull` → install
  the package → run the relevant script with its config.
- In Colab, `/content` is wiped when the runtime resets. Keep the data a GPU job needs,
  checkpoints, and its run folder on Google Drive (`MyDrive/proctor-assist/`). Copy small result
  files (metrics, per-epoch history, config) back into the repo and commit them.
- Long jobs in Colab: start them in the background with logs on Drive
  (`nohup python -m ... > <drive>/logs/<run_id>.log 2>&1 &`), then poll the log instead of
  blocking one cell. Save a checkpoint every epoch so a disconnect does not lose the run.
- Before running a cell that installs packages, deletes files, or starts a GPU job longer
  than a few minutes, state what will run and wait for approval.
- Record the GPU type (`nvidia-smi`) and library versions in every run folder, because Colab
  can assign different GPUs between sessions.
- Kaggle and GitHub tokens live in Colab Secrets, or locally in `~/.kaggle/access_token` or an
  environment variable. Never print them or write them to code, notebooks, logs, or the repo.

## Code conventions (defaults, can be changed by agreement)

- Python package in `src/`, experiments driven by config files, core logic outside notebooks.
- Every run writes a folder with its config, metrics, and (for trained models) per-epoch
  history of train/val loss and train/val metric.
- Fixed seeds; library versions and GPU type recorded per run.
- The app reuses the evaluation pipeline code instead of a copy.
- Tests for data splitting, window extraction, metrics, and perturbations.
- Figures are generated by scripts from saved results.

## How to work with me

- The goal is the anchor, and the plan is a first hypothesis. If reality differs from the
  plan, adapt the steps, check the result, and record the reason in `docs/decisions.md`.
  Goals, scope, and the rules in this file do not change.
- Work one milestone of `docs/PLAN.md` at a time.
- For small implementation choices that are easy to undo, decide yourself. Explain the
  reason in one or two sentences, and record the choice in `docs/decisions.md`.
- Ask the owner first for:
  - owner decisions: the class mapping, labels and ground truth, and changes to the
    experiment design (PLAN §5).
  - large design choices with real trade-offs. Give the options and a recommendation.
  - actions that are hard to undo: package installs in Colab, deleted files, GPU jobs longer
    than a few minutes, commits, and pushes.
- Owner decisions stay in the chat (D16). Ask each one as a multiple-choice question. Give a
  recommendation, and the gain and the cost of each option.
- Ask in dependency order: a decision that other decisions depend on comes first. Say what
  each question depends on and what it changes. If an answer makes an earlier answer invalid
  or conflicts with the plan, stop and say so. Answers are provisional until the owner
  accepts a final recap.
- For each check that the owner must do, say what to open and what to look at. Say which
  result is expected, and what each result means for the decision.
- Check facts against a primary source before you use them. If a fact is not checked yet,
  write "not verified" or `TBD`.
- If something blocks the work, ask the owner in one message. Otherwise, keep working.
- Explain non-obvious design choices so the owner can defend them in the report and
  presentation.
- After finishing, run the tests, list what changed, and note decisions in `docs/decisions.md`.
- Prefer small commits with clear messages.
- If a library API differs from what you expect, check the installed version's docs
  instead of guessing.
