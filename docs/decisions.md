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

## D11. OEP download: webcam subset, renamed videos, committed manifest (2026-11-10)

- Decision (owner): download only `READ_ME.txt`, the 24 `gt.txt` files, and the 24 webcam
  videos from the Kaggle mirror, not all 12.08 GB. The script is
  `python -m proctor_assist.download`.
- Reason: audio and the wearcam are out of scope (PLAN §2). The full download is one zip, so
  the zip and its extracted files must fit on the disk at the same time. The drive of the
  repository does not have that space free. Also, less personal data is stored.
- Layout: `ROOT/data/raw/oep/` holds `READ_ME.txt`, `subjectN/gt.txt`, and
  `subjectN/webcam.avi`. `READ_ME.txt` names the webcam video `<username>1.avi`. The rename
  keeps the usernames of the subjects out of the repository, the logs, and the app. The file
  content does not change.
- Manifest: `manifests/oep_raw.sha256` (`sha256sum` format) is committed. The first run writes
  it. Each later run, on any machine or in Colab, compares its files with it and fails on a
  difference. This is the check that D8 asks for.
- Source: the Kaggle mirror is the only source that is still reachable. On 2026-11-10, the
  original MSU download link redirected to the department homepage, and `cvlab.cse.msu.edu`
  refused connections. The archived MSU dataset page (Wayback Machine, 2024) asks for a
  citation of the paper and names no license.
- Rejected: download all files. Rejected: keep the original file names. Rejected: a manifest
  only in the data folder, because then the local copy and the Drive copy have no shared
  reference.

## D12. Finding: meaning of the type codes in `gt.txt` (2026-11-10)

This entry is a finding, not a choice. The class mapping is a separate owner decision. A draft
is in `docs/class_mapping_PROPOSAL.md`.

- Source: Atoum et al., "Automated Online Exam Proctoring", section IV and Fig. 9. The copy is
  the authors' version from the MSU lab site, archived by the Wayback Machine on 2023-01-06
  (`web.archive.org/web/20230106084226/http://cvlab.cse.msu.edu/pdfs/Atoum_Chen_Liu_Hsu_Liu_OEP.pdf`,
  15 pages). Its page header says "accepted with minor revision, Dec 30, 2015". The final IEEE
  version is not verified. The snapshot of 2025-01-18 is cut off at 1 MiB and cannot be read.
- The paper defines 5 types: (1) cheating from a book, notes, or text on paper, (2) talking
  to a person in the room, (3) using the Internet, (4) asking a friend over the phone,
  (5) using a phone. The caption of Fig. 9 says "use a phone or other devices" for type 5.
- The files do not use the numbers 1 to 5. They hold the codes 1, 2, 3, 5, and 6
  (`results/eda/summary.json`). No source that we found explains code 5, code 6, or the
  missing code 4. `READ_ME.txt` gives no code list.
- Reading of the codes, with the evidence for each:

  | Code in `gt.txt` | Meaning | Evidence |
  |---|---|---|
  | 1 | Paper type 1: text on paper | Paper, and the share of instances matches Fig. 9 |
  | 2 | Paper type 2: talk to a person in the room | Paper, and the share matches Fig. 9 |
  | 3 | Paper type 3: use the Internet | Paper, and the share matches Fig. 9 |
  | 5 | Paper type 4: ask a friend over the phone | Inferred. Share matches Fig. 9, and sample frames |
  | 6 | Paper type 5: use a phone | Inferred. Share is near Fig. 9, and sample frames |

- Evidence from the data (`results/eda/summary.md`): the shares of instances for the codes
  1, 2, 3, 5, 6 are 34.0%, 50.5%, 9.2%, 2.7%, and 3.5%. Fig. 9 gives 34%, 50%, 10%, 3%, and
  4% for the types 1 to 5. The shares of labeled seconds are 20.1%, 50.8%, 21.3%, 1.5%, and
  6.3%. Fig. 9 gives 16%, 50%, 21%, 1%, and 12%. The order agrees. The values for code 1 and
  code 6 differ from the paper.
- Evidence from frames (AI observation, the owner must verify it): in the contact sheet of
  code 5 (`data/eda/frames/code5.jpg`), 7 of the 8 frames show a phone at the ear. In the
  contact sheet of code 6, no frame shows a phone at the ear, and most subjects look down. The
  webcam does not see the desk, so the phone itself is not visible.
- Time format: MMSS from the start of the video. With this reading, the last label of all 24
  subjects is inside the video. If a time were plain seconds, this holds for 0 subjects. The
  paper says that the labels have a resolution of one second.
- Differences from the paper that stay open: the files hold 565 rows and the paper reports
  569 instances. The total length is 7181 s as end - start, or 7745 s with the end second
  included. The paper reports 7235 s. Intervals cover 26.7% of the video time. The paper says
  "nearly 20%". The Kaggle mirror can differ from the data that the paper used.
- The videos agree with the paper: 640 x 480 at 25 fps, with a mean length of 17.5 minutes.
- The paper's own classifier used other groups: class 1 for text, class 2 for all speech
  (types 2 and 4), class 3 for phone or laptop. A separate module detected type 3 from the
  number of open windows. That system had audio and a wearcam. This project has neither.
- Still to verify (owner): the type list in the final IEEE version (IEEE Xplore through
  BINUS), and the contact sheets. The dataset author can confirm the codes 5 and 6. His
  address is in `READ_ME.txt`.

## D13. Label parser and EDA (2026-11-10)

- Parser (`proctor_assist.labels`): it ignores spaces and tabs around a field. A line that
  breaks the format stays in the output with a reason, and gets no values. The parser never
  repairs a label. Reason: labels are an owner decision, and a silent repair hides a data
  problem. The raw files are never edited, because they must match the manifest (D11).
- EDA (`python -m proctor_assist.eda`): one script writes all tables, `summary.md`, and the
  figures to `results/eda/`. Documents take their numbers from these files.
- Committed: counts and totals, and the few rows that have a problem. Not committed
  (`data/eda/`): the contact sheets, because they show faces, and the full list of overlapping
  rows, because the license of the labels is unknown.
- Interval length in the EDA: end - start. It is not known if the annotators meant the end
  second as part of the interval. `summary.json` has both totals. The convention for the
  ground truth is an owner decision (see the proposal).
- Video properties come from the file header through OpenCV. The script also reads the last
  frame of each video as a check. It does not decode every frame, so the frame count is the
  count from the header. Milestone 3 decodes the frames. If the counts differ, that step
  shows it.
- Dependency: `opencv-python`, approved by the owner on 2026-11-10 and not pinned (D5).
  Milestone 3 needs it again for frame extraction. `summary.json` records the local version.
- Figures: one fixed color per type code. A validator checked the colors for color-blind
  readers. The timeline also gives each code its own lane, so color is not the only cue.
- Rejected: a reader for the AVI header from the standard library. It needs no dependency,
  but it cannot give sample frames. Rejected: a repair of bad lines in the parser.
