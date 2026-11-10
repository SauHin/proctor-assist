# Decisions

Implementation decisions made with the owner. `docs/PLAN.md` holds goals, scope, and
constraints. Each entry gives the decision, the reason, and the rejected alternatives.

D1 to D8 come from the setup work for the previous topic and still apply. They were adapted
to this project on 2026-11-10.

## D1. Public GitHub repository (2026-09-29, adapted 2026-11-10)

- Decision: the repository is public at `SauHin/proctor-assist`.
- Reason: Colab can clone without a token, so no GitHub token is stored in Colab Secrets.
- Consequence: never commit Kaggle data, videos, frames, landmark files, checkpoints, or
  notebook outputs. `.gitignore` blocks `data/`, `runs/`, `logs/`, `models/`, and checkpoint
  files.
- Rejected: private repository with a read-only token in Colab Secrets.

## D2. Workflow between git, Colab, and Drive (2026-09-29)

- Decision: edit locally, commit, and push to `main`. In Colab, pull, run
  `pip install -e ".[dev]"`, and start scripts with `!python -m proctor_assist.<module>`.
- The notebook `notebooks/colab.ipynb` only mounts Drive, clones or pulls, installs, and
  runs commands. Core logic stays in `src/`.
- Scripts run in a subprocess (`!python -m`), so a new editable install is visible without a
  kernel restart.
- Work directly on `main` with small commits, so Colab pulls one branch only.

## D3. Storage on Google Drive (2026-09-29, adapted 2026-11-10)

- Decision: `MyDrive/proctor-assist/` holds `data/raw`, `data/processed`, `runs/`, and
  `logs/`.
- Run folder name: `<name>_<UTC YYYYMMDD-HHMMSS>_<git sha>`. Each run folder has `env.json`
  with the GPU, `nvidia-smi` output, Python version, installed packages, git sha, and a
  flag for uncommitted code. A new run never overwrites an existing folder.
- Checkpoints: keep `last` and `best` only. Per-epoch history stays complete in the history
  file.
- Rejected: keep a checkpoint for every epoch. Checkpoints of video models are large, and
  Drive quota is limited.

## D4. Result files back to the repository (2026-09-29)

- Decision: copy small result files from Drive to `results/` with Google Drive for Desktop,
  from milestone 4. The owner installs Drive for Desktop before then.
- Reason: the copy is mechanical. No number passes through an AI transcription.
- Rejected: Colab pushes results itself, because it needs a write token and gives commit
  conflicts. Rejected: copying numbers from cell output.

## D5. Dependencies (2026-09-29)

- Decision: add a dependency to `pyproject.toml` at the milestone that needs it. Do not pin
  packages that Colab preinstalls, such as torch and numpy.
- Reason: a reinstall of torch is slow and can break the CUDA match. `env.json` records the
  exact version of every package per run, so each result stays traceable.
- Rejected: a full lock file, because it conflicts with changes to the Colab base image.

## D6. Two test layers (2026-09-29, adapted 2026-11-10)

- Decision: unit tests without data or GPU (window extraction, metrics, perturbations,
  split logic on toy data) run locally and in Colab. Data tests (no subject in two splits,
  split sizes) need the data folder. If the data folder is not present, the data tests skip.
- The full `pytest` suite must pass in Colab before an experiment run.
- Local test results are a development aid only. They are not evidence in the report.
- Rejected: Colab only, because each small change then needs a live Colab session.
  Rejected: GitHub Actions, because it adds a workflow file and gives feedback only after a
  push.

## D7. Configuration format (2026-09-29)

- Decision: YAML configuration files, from the first experiment (milestone 4).
- Reason: YAML allows comments, so each hyperparameter can carry its reason for the report.

## D8. Local first, Colab for GPU jobs only (2026-10-12, adapted 2026-11-10)

- Decision (owner): work that needs no GPU runs locally in `.venv`: data prep, EDA, CPU
  evaluation, and all tests. Only GPU jobs run in Colab: model training, and feature
  extraction that is too slow on CPU.
- Reason: a Colab session is a slow loop for small steps, and the Colab MCP connection is
  not always available.
- Storage: without `PROCTOR_ROOT`, data and run folders live in the repo's gitignored
  `data/` and `runs/`. In Colab, they live on Drive (`MyDrive/proctor-assist/`). `env.ROOT`
  picks the location.
- This replaces the Colab-only parts of D2, D3, and D6.
- The local raw data must be the same files as on Drive. A manifest with SHA-256 hashes
  checks this.
- Open for milestone 4: how the data for a GPU job gets to Drive. Option A: set
  `PROCTOR_ROOT` to a Google Drive for Desktop folder, so local and Colab share one folder.
  Option B: upload the processed data by hand before each GPU job.

## D9. Topic: cheating detection for online exam proctors (2026-11-09)

- Context: the lecturer did not approve the previous topic, a semantic resume matcher. As
  the owner understood it, the topic was too general, and general AI tools can solve it.
  The lecturer asked for a computer vision project based on papers. Datasets from existing
  papers are allowed, and real user testing is not required.
- Decision (owner): detect cheating moments in webcam video of online exams, with the OEP
  dataset and a proctor dashboard.
- Reason: a live webcam demo, a public dataset with time-stamped labels, and prior deep
  learning work. The prior work gives a proven method to reproduce (P1) and clear gaps to
  test: temporal context, landmarks, event-level metrics, and CPU scalability.
- Rejected after a comparison with the owner:
  - ForestNet (drivers of forest loss in Indonesia). It has the strongest data but no live
    demo. A proposal draft exists as a Claude doc.
  - Wildfire smoke detection (FIgLib). Its license is not stated.
  - WHO hand-washing steps, HELMET (motorcycle helmet use), RDD2022 (road damage), TBX11K
    (tuberculosis X-rays), MARIDA (marine debris), rip currents, Landslide4Sense,
    Coralscapes, and SeaTurtleID2022.
- Accepted weaknesses: a small dataset with mostly acted cheating, the license "Unknown" on
  Kaggle, and the ethics and regulation points in PLAN §9.
- This repository started fresh on 2026-11-10. The code of the previous topic is not part
  of it.

## D10. Fixed experiment variables, flexible architectures (2026-11-09)

- Decision (owner): the variables in PLAN §5 are fixed. The architecture inside a scenario
  can change, for example another per-frame CNN for P1-style models or another video model
  for E3.
- Reason: after a model change, the comparison stays valid, and the report can still
  explain each choice.
- Rule: every model gives one suspicion score from 0 to 1 per window. Thus a model change
  needs no change to the app. Each change of architecture gets an entry here with its
  reason.
- P1 keeps the configuration of the reference paper, because P1 is the proven method and
  the app fallback.
