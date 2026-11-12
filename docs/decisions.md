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
- The paper defines 5 types. The caption of Fig. 9 says "use a phone or other devices" for
  type 5.
  1. Cheating from a book, notes, or text on paper.
  2. Talking to a person in the room.
  3. Using the Internet.
  4. Asking a friend over the phone.
  5. Using a phone.
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
- Evidence from frames, as an AI observation. The owner checked the sheets later (D14). In
  the contact sheet of code 5 (`data/eda/frames/code5.jpg`), 7 of the 8 frames show a phone
  at the ear. In the
  contact sheet of code 6, no frame shows a phone at the ear, and most subjects look down. The
  webcam does not see the desk, so the phone itself is not visible.
- Time format: MMSS from the start of the video. With this reading, the last label of all 24
  subjects is inside the video. If a time were plain seconds, this holds for 0 subjects. The
  paper says that the labels have a resolution of one second.
- Differences from the paper that stay open: the files hold 565 rows and the paper reports
  569 instances. The total length is 7181 s as end minus start, or 7745 s with the end second
  included. The paper reports 7235 s. Intervals cover 26.7% of the video time. The paper says
  "nearly 20%". The Kaggle mirror can differ from the data that the paper used.
- The videos agree with the paper: 640 x 480 at 25 fps, with a mean length of 17.5 minutes.
- The paper's own classifier used other groups. Class 1 is text. Class 2 is all speech
  (types 2 and 4). Class 3 is phone or laptop. A separate module detected type 3 from the
  number of open windows. That system had audio and a wearcam. This project has neither.
- Still to check (owner): the type list in the final IEEE version (IEEE Xplore through
  BINUS), and the contact sheets. The dataset author can answer the question about the codes
  5 and 6. The address is in `READ_ME.txt`.

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
- Interval length in the EDA: end minus start. No source says whether the end second is part of
  the interval. `summary.json` has both totals. The convention for the
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

## D14. Class mapping and label cleaning (decided by the owner in chat, 2026-11-10)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record. The options that the
owner did not choose are in `docs/class_mapping_PROPOSAL.md`.

- Classes (option A):

  | Class | Name | OEP codes |
  |---|---|---|
  | 0 | no cheat | no label |
  | 1 | looks at an object | 1 and 6 |
  | 2 | talks to a person | 2 |
  | 3 | phone call | 5 |

- Code 3 (Internet): masked. A masked second is left out of training and evaluation. It is
  not labeled "no cheat". Where code 3 shares a second with code 1 or 2, the visible code
  wins.
- subject7 line 14 (`0900 00903 6`): read as `0900 0903 6`.
- subject7 duplicate rows: keep one copy.
- subject8 line 3 (`0112 1019 3`): kept as it is.
- subject1 line 6 (`0230 0336 2`): corrected to `0230 0236 2`.
- Classes that share a second: the rare class wins. The order is 3, then 1, then 2.
- End second: an interval is [start, end).
- Checks that the owner reported in the chat:
  - `code5.jpg`: a phone is at the ear in 5 or more of the 8 frames.
  - `code1.jpg` and `code6.jpg`: the owner chose the option for sheets that look alike.
  - `check_subject1_0220_0340.jpg`: the owner chose the option with this description. A
    second person is in the frame only from about 02:27 to 02:39. From 03:02 to 03:36, the
    subject is alone.
- Not checked: the final IEEE version of the paper. Also, subject8 line 3 cannot be checked
  from the webcam, because Internet use is not visible.
- Reasons, from the options that the owner chose:
  - Classes: the project uses the webcam only, so the webcam must be able to tell each class
    apart. The class count equals the reference paper, and phone use stays a cheating type
    (PLAN §2). The cost: class 1 is wider than "read text" in the paper.
  - Code 3: in the webcam, Internet use looks like normal exam work. The visible code wins,
    so that visible behavior is not lost.
  - subject8 line 3: with code 3 masked, a wrong long row only removes data. It adds no wrong
    label.
  - subject1 line 6: the row is then in line with the frame sheet. The cost: the label
    changes on an inference. If the subject talked to a person outside the frame, those
    seconds are now "no cheat".
  - Rare class wins: no data is lost, and the rare classes keep their frames. The cost: one
    of two true labels is dropped.
  - [start, end): the effect is at most 1 s per interval. PLAN §6 plans a tolerance of a few
    seconds at event boundaries.
- Consequences: the splits of D15 stay valid, and D10 does not change.

## D15. Subject splits for protocols A and B (2026-11-10)

- Decision (owner): the 4 actors with code 5 (phone call) are spread over the splits, so that
  each test set has one of them. In protocol B, the fold after the test fold is the
  validation set.
- File: `python -m proctor_assist.splits` writes `splits/subjects.json` with seed 0. The file
  is committed, and every later step reads it. Thus the local machine and Colab use the same
  subjects.
- Leakage test: a test fails on any subject in more than one split. It also compares the file
  with the output of the code, so a subject that was moved by hand fails the test.
- Protocol A: 11 actors train, 3 actors validate, and the test set is the 9 real exam takers
  and 1 actor. The test actor is a random pick from the 4 actors with code 5.
- Protocol B: 4 folds of 6 subjects. The subjects are dealt like cards in three groups: the
  actors with code 5, the other actors, and the real exam takers. Each fold gets 1 actor with
  code 5 and 2 or 3 real exam takers. For each test fold, 12 subjects train and 6 validate.
- Reason for the spread: the real exam takers have no code 5. Without the spread, a test set
  can have no phone call at all. Then the F1 of that class has no value, and macro F1 is not
  comparable between folds.
- This is not data leakage. The split uses only the fact that a subject has code 5. It uses
  no model result, and it is fixed before any frame is extracted. The cost: the split is not
  fully random. The report must say so.
- AI choices inside this decision, easy to change before milestone 4:
  - Protocol A has 3 validation actors, about one fifth of the 14. The validation split of
    the reference paper is not verified.
  - One validation actor in protocol A also has code 5. Then protocol A has the same pattern
    as protocol B: 2 of these actors train, 1 validates, and 1 tests. The cost: only 2 of
    them are in the training set.
- Known limits (`results/splits/composition.csv`): the test set of protocol A has 2 intervals
  of code 5 and 1 interval of code 6. In protocol B, the test set of fold 4 has no interval
  of code 6. If the owner makes code 6 a class of its own, the folds need a second look.
- The split uses Python's `random` module with a fixed seed. The committed file is the
  reference. If the test fails on another Python version, compare the file with the new
  output before any change.
- Rejected by the owner: a random split by group only, and 3 validation subjects from the
  training data of each fold. Rejected: `StratifiedGroupKFold` from scikit-learn. The project
  does not need that dependency yet, and it cannot spread two properties at once.

## D16. Owner decisions are made in the chat, and the AI records them (2026-11-10)

- Decision (owner): the owner makes each decision in the chat, from a multiple-choice
  question. Each question has a recommendation, and the gain and cost of each option. The AI
  then records
  the answer in this file, marked as decided by the owner in chat.
- This changes one integrity rule in `CLAUDE.md`. Before, the owner had to type the final
  class mapping into the file. Now the owner decides, and the AI types.
- Reason (owner): the workflow stays in one place. The owner was afraid that one answer
  changes the whole chain of experiments without notice.
- Safeguards:
  - Questions come in dependency order. An answer that makes an earlier answer invalid stops
    the round.
  - Answers are provisional until the owner accepts a final recap.
  - The AI never picks an option, and never records a check that the owner did not report.
  - The AI usage log states that the AI wrote the options and typed the record.
- Rejected: the AI gives a filled block and the owner pastes it into the file. It keeps the
  old rule, but it adds a manual step outside the chat for each label decision.

## D17. Labels per second from D14 (2026-11-10)

- `proctor_assist.labels` holds the values of D14 as constants: the class names, the map
  from code to class, the masked code, the class order, and the two corrected rows.
- `clean_labels` applies the corrections and removes the duplicate rows. It fails on a row
  with a problem that D14 does not cover. If a corrected row in `gt.txt` is not the text that
  D14 names, it also fails. Thus a correction cannot land on a wrong row.
- `second_labels` gives each second of a video one value: 0, a class from 1 to 3, or -1 for
  a masked second.
- The EDA still reads the raw rows, so `results/eda/` describes the files as they are.
- Reason for constants in the code: the values change only by an owner decision, and a test
  checks each rule of D14. A YAML file (D7) starts with the first experiment.

## D18. Frame extraction and windows (decided by the owner in chat, 2026-11-10)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record.

- Frame rate: 5 fps. The code keeps every 5th frame of the 25 fps video.
- Stored frame: the whole image, resized to 224 x 224, with no crop. The P1 paper uses the
  same input size (checked on the MDPI page of the paper, section 3.4).
- Window target: the D14 label of the last second of the window. All models answer the same
  question: is this second cheating?
- Window length for E1, E2, and E3: 3 s (15 frames), with one window per second.
- A window is formed when the data is read, not at extraction. Thus both window choices are
  cheap to change.
- Open until milestone 4 (D8): where the processed data lives for Colab, and how it gets
  there.
- Reasons, from the options that the owner chose:
  - 5 fps: a window of 2 to 4 s holds 10 to 20 frames, which is enough for the temporal
    models. The per-frame models can use a part of the frames. One extraction serves all
    models. The cost: 5 times more frames than at 1 fps.
  - 224 x 224, whole image: the smallest option, and no resize is necessary during training.
    A second person at the edge of the frame stays visible. The cost: the 4:3 image is
    squeezed a little.
  - Last second as target: the per-frame models and the temporal models get the same target.
    Only the input differs, so the comparison of P1 with E1 is fair. No short cheating span
    is lost. The cost: at the start of an event, most of the window still shows normal work.
  - 3 s: the middle of the range in PLAN §5. The cost: more CPU work per window than 2 s.
- Options that the owner did not choose: 1, 2, or 25 fps. Frames of 320 x 240 or 640 x 480.
  A target from the majority of the seconds, or from any cheating second. Windows of 2 or
  4 s.
- The questions gave sizes that were estimates from 2 subjects. The real numbers are in
  `results/frames/`, from the extraction run.
- Not in the P1 paper: the paper gives no frame rate. It reports 9000 training frames and
  7000 test frames.

## D19. Extraction and window code (2026-11-10)

These are AI choices inside D18. They are easy to change.

- Files: `data/processed/frames/subjectN/000000.jpg` and so on. Frame k is at k / 5 seconds.
  Second t has the frames 5t to 5t + 4.
- The JPEG quality is 90. The resize uses area interpolation.
- One function, `frames.preprocess`, does the resize. The app must use the same function
  (PLAN §7).
- The script writes a marker file for each subject as its last step. After an interruption,
  the next run extracts that subject again.
- If `splits/subjects.json` does not cover all subjects, the script stops. Thus no frame is
  extracted before the split.
- A subject has `frames_saved // 5` whole seconds. A partial second at the end is dropped.
- A window ends at second t and holds 15 frames. Seconds 0 and 1 have no full window before
  them, so they give no window. There is no padding.
- The mask applies to the target only. A masked second gives no window. Its frames can
  still be input for a later window.
- `results/frames/summary.csv` gives the number of seconds per subject to the window code.
  Thus Colab does not need to count files.
- Result of the run (`results/frames/summary.json`): 125706 frames, 1.449 GB, and 25132 whole
  seconds. For all 24 videos, the decoded frame count equals the count in the header. This
  closes the open point of D13.
- `results/windows/summary.csv` gives the windows per class for each part of each split. The
  classes are not balanced. Milestone 4 must handle this.
- Rejected: frames as arrays without compression. One 224 x 224 frame with 3 channels is
  about 150 KB, and the JPEG files are about 12 KB per frame. Rejected: decode from the video
  during training. Each window then needs a seek in the video and a decode.

## D20. Frames go to Colab as zip files through Drive for Desktop (decided by the owner in chat, 2026-11-10)

The owner chose this from one multiple-choice question in the chat (D16). It closes the open
point of D8.

- Decision (owner): `python -m proctor_assist.pack pack` writes one zip file per subject into
  the Google Drive folder of the project. Drive for Desktop uploads the files. In Colab,
  `pack unpack` opens them on the local disk of the runtime.
- Check: `manifests/frames_zip.sha256` is committed. If a zip file differs from the manifest,
  `unpack` stops. Thus Colab trains on the same bytes that the laptop extracted.
- Reasons, from the option that the owner chose: the data is the same on both machines,
  there is no manual upload, and the training reads from a fast local disk. The cost: one
  upload of 1.449 GB, and the same amount of Google Drive storage.
- Options that the owner did not choose: Colab extracts the frames from the Kaggle files
  itself. All data lives in the Drive folder through `PROCTOR_ROOT`.
- AI choices inside this decision:
  - `env.FAST_DATA` is the folder for the frames. On the laptop, it is `data/`. In Colab, it
    is `/content/proctor-data` on the local disk. A runtime reset empties it, and `unpack`
    fills it again.
  - The zip files have no compression, because JPEG is compressed already.
  - Each file in a zip has a fixed date. Thus the same frames give the same zip bytes.
- Colab MCP: the connection test passed on 2026-11-10. The server connected to a notebook
  and ran a cell.
- Check in Colab on 2026-11-10, on a GPU runtime: `unpack` accepted all 24 zip files. The
  frame count of each subject equals `results/frames/summary.csv`. The setup check wrote
  `runs/setup-check_20261110-124207_d008233/env.json` on Drive, and `pytest` passed (the 3
  tests that need the raw videos were skipped).
- Limits that this test showed:
  - A tool cannot mount Google Drive. The owner must run the mount cell in the Colab tab and
    allow the access.
  - A change of the runtime type empties the local disk and drops the tool connection. Then
    the clone, the install, and `unpack` must run again.
  - In the Colab GPU runtime, `cv2` is version 4.14.0. The laptop has 5.0.0. The frames come
    from the laptop, so the extraction is the same. The JPEG decode can differ a little.
    Milestone 4 must measure this before the first training run.

## D21. P1 design (decided by the owner in chat, 2026-11-10)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record.

- Backbone: MobileNetV2 with pretrained weights. All layers are fine-tuned.
- Frames: training uses all stored frames, 5 per second. The score of a second comes from
  its last frame.
- Augmentation: the types of the paper, with realistic sizes. Rotation up to 15 degrees,
  shift and zoom up to 10%, and horizontal flip. No vertical flip. This is a deviation from
  the paper (D10).
- Class imbalance: balanced sampling, as in the paper. Each class has the same chance in a
  batch.
- From the paper, with no change (section 3.4 and Table 1 of the paper):
  - Input of 224 x 224, and softmax over 4 classes.
  - Adam, with a learning rate of 0.0001 and a batch size of 16.
  - At most 20 epochs, and early stopping with patience 10.
  - 5 dense layers.
- Different from the paper by a project rule: the validation set is 3 held-out subjects
  (D15). The paper used 5-fold cross-validation on frames. Frames of one subject are then in
  the training part and in the validation part. Also, the classes are those of D14.
- AI choices where the paper gives no value. They are easy to change (D10):
  - Head: a 3 x 3 convolution with 100 filters, then max pooling 2 x 2, then flatten to 400
    values. After that come 5 dense layers of 128 units with ReLU, and the output layer. The
    sizes 5 x 5 x 100 and 400 in the paper fit this order of layers.
  - Early stopping looks at the validation loss.
  - The framework is PyTorch, with the ImageNet weights from torchvision.
- Reasons, from the options that the owner chose:
  - Fine-tune: the features can adapt to head poses and to a phone at the ear. The cost:
    each training run needs a GPU, and 11 training subjects give a risk of overfitting.
  - All frames for training, one frame for the score: the rare classes get 5 times more
    examples. P1 still decides from one frame, so the comparison with E1 stays clean.
  - Realistic augmentation: a webcam never gives a face that is upside down.
  - Balanced sampling: the model cannot win with "no cheat" for each frame. The cost: the
    same phone call frames come back often.
- Options that the owner did not choose: a frozen backbone. Training on 1 frame per second.
  A score from the mean of 5 frames. The exact augmentation of the paper, or none. Class
  weights in the loss, or no handling of the imbalance.
- Facts from the paper for the comparison (MDPI page, Tables 2 to 4):
  - 84.52% is the macro F1 of the best of 5 folds. The mean of the 5 folds is 80.81%.
  - The source of these scores is not clear in the paper: the test set, or the validation
    folds.
  - Table 4 names the class "5 (Call Friend)". This supports the reading of code 5 in D12.
  - The paper gives no dense units, no augmentation sizes, and no frame rate.

## D22. Training code for the per-frame models (2026-11-10)

These are AI choices inside D21. They are easy to change.

- One command trains a model from a YAML file (D7): `python -m proctor_assist.train
  configs/p1_a.yaml`. The YAML file holds each value of D21 with a comment on its source.
- Run folder (D3): `config.yaml`, `env.json`, `history.csv` with one row per epoch, `last.pt`,
  `best.pt`, `metrics.json`, and the class probabilities per second for the validation and
  test subjects. The scores per second are the input for the event metrics of milestone 6.
- After a disconnect, `--resume <run folder>` continues the run from `last.pt`.
- One epoch draws as many frames as the training table has. With balanced sampling, a frame
  of a rare class is drawn many times, and a frame of class 0 is not always drawn.
- The validation loss and all metrics use one frame per second, without balanced sampling.
  Thus they show the data as it is.
- All models are scored on the same seconds: those with a full window of 3 s before them.
- `metrics.json` also has a baseline that gives the largest training class for each row. A
  trained model must beat it (PLAN §5).
- The frames are read with torchvision, not with OpenCV. `python -m proctor_assist.data`
  prints one hash of the decoded pixels of 72 frames. The hash on the laptop and the hash in
  Colab must be equal before a training run. This is the check that D20 asks for.
- ImageNet weights: `IMAGENET1K_V1` of torchvision, the original weights of MobileNetV2.
- Dependencies in `pyproject.toml`, not pinned (D5): torch, torchvision, scikit-learn, and
  pyyaml. The laptop got torchvision 0.29.2, which matches its torch 2.14.1.
- Labels in Colab: `pack` also copies the 24 `gt.txt` files into the Drive folder. `unpack`
  checks them against `manifests/oep_raw.sha256`. Reason: the training builds its targets
  from `gt.txt`, and the labels do not go into the public repository.
- `--smoke` runs 2 short epochs on a few rows with no pretrained weights. It only shows
  that the pipeline works. Its numbers are not results.

## D23. Finding: P1 on protocol A does not transfer to new subjects (2026-11-10)

This entry is a result and a diagnosis, not a choice. All numbers come from
`results/p1_a_20261110-132143_edfc724/`.

- Run: code at `edfc724`, GPU NVIDIA L4, 12 epochs of about 119 s, 1426 s in total. Early
  stopping kept the model of epoch 2.
- Training curve (`history.csv`): the training loss falls from 0.305 to 0.087. The validation
  loss is lowest at epoch 2 (0.923), and then it rises to 1.268. The model overfits from the
  second epoch.
- Scores of the kept model (`metrics.json` and `metrics_train.json`):

  | Subjects | Seconds | Macro F1 | Baseline macro F1 | Accuracy | Baseline accuracy | ROC-AUC cheat |
  |---|---|---|---|---|---|---|
  | Training (11 actors) | 9208 | 0.767 | 0.227 | 0.892 | 0.832 | 0.978 |
  | Validation (3 actors) | 2733 | 0.387 | 0.224 | 0.804 | 0.813 | 0.501 |
  | Test (9 real, 1 actor) | 11742 | 0.423 | 0.210 | 0.693 | 0.726 | 0.631 |

- Diagnosis: the same scoring steps give a ROC-AUC of 0.978 on the training subjects. Thus
  the low scores are not an error in the scoring code. The model learned the training
  subjects, and it does not transfer to new people.
- Sanity check of PLAN §5: the loss decreases, and macro F1 is above the baseline. But the
  accuracy is below the baseline on validation and on test. The ROC-AUC on validation is at
  chance level.
- Per subject (`per_subject.csv`): the ROC-AUC goes from 0.318 to 0.824 on the validation
  subjects, and from 0.367 to 0.871 on the test subjects. The result depends on the person.
- Per type code (`per_code.csv`), validation subjects: the suspicion is above 0.5 for 25.6%
  of the seconds of code 1, 44.9% of code 2, 66.7% of code 5, and 29.5% of code 6. Code 6 is
  not worse than code 1. Thus the merge of D14 is not the cause.
- Comparison with the paper: the paper reports 84.52%, from 5-fold cross-validation on
  frames. There, frames of one subject are in the training part and in the validation part.
  Our score on the training subjects shows how high a score is when the model knows the
  person. This is a likely reason for the difference. It is not proven.
- Limits: this is one run with one seed. The validation set has 3 subjects, so each number
  on it is unstable.
- Tool: `python -m proctor_assist.evaluate` writes the tables per subject and per type code,
  and scores the training subjects from a checkpoint.
- The next step for P1 is in D24.

## D24. P1 runs with no change on protocol B (decided by the owner in chat, 2026-11-10)

The owner chose this from one multiple-choice question in the chat (D16), after the result
of D23.

- Decision (owner): P1 runs on the 4 folds of protocol B with the same values as on protocol
  A (D21).
- Reasons, from the option that the owner chose: protocol B is the main protocol, so these
  runs are necessary. Nothing is tuned. The runs show if the failure of D23 is general, or
  special to protocol A.
- Options that the owner did not choose:
  - A test of a frozen backbone on protocol A first. It is cheap, but 3 validation subjects
    are too few to judge a change.
  - Accept the result and go on to R0.
- A later change to P1 is judged on validation subjects only, never on test subjects. The
  result of D23 stays in the repository and in the report.
- Code: `configs/p1_b.yaml` has the values of `configs/p1_a.yaml`, and a test checks this.
  `--fold` picks the fold. The run folder gets the values that the run used, with the fold.

## D25. R0 design (decided by the owner in chat, 2026-11-10)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record.

- Signal: head yaw and pitch from the MediaPipe face landmarker, on the original video of
  640 x 480. The rule uses one frame per second, the same frame as the score of P1.
- Normal position: the median yaw and pitch of the same subject, from the start of the video
  to the current second.
- Score: the angle distance from the normal position, divided by 30 degrees, and cut at 1.
  The 30 is an AI choice. It only scales the score, and it was not tuned on any subject.
- No face in a second: the score of the last second with a face. Before the first face, the
  score is 0.
- No training. The alarm threshold comes from validation subjects at milestone 6.
- R0 gives one suspicion score, not 4 classes. The comparison with P1 uses the ROC-AUC and
  the event metrics.
- Reasons, from the options that the owner chose:
  - Own median: the normal head position differs between people. The rule looks only at the
    past, so it works live, and it uses no labels. The cost: the rule needs some time at the
    start. If a subject looks away for more than half of the time, the normal position
    moves.
  - Last score for a second with no face: a high score stays if the face is lost in a turn,
    and a low score stays if it is lost in the normal position. The cost: the score can
    stay at an old value for a long time.
- Options that the owner did not choose: a fixed normal position that faces the camera. A
  score of 1 or of 0 for a second with no face.
- Model file: `face_landmarker.task` from the MediaPipe model storage of Google, 3758596
  bytes, SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. It is in
  `data/models/`, which is not committed. The code checks the hash.
- Dependency: mediapipe 1.2.2 on the laptop, as the optional extra `landmarks`. Colab does
  not install it. It also installed opencv-contrib-python 5.0.0.93 next to opencv-python.
- Note for the report: the first look at the angles used 2 training actors of protocol A
  (subjects 1 and 3). In protocol B, each subject is a test subject one time. The values
  from that look were used to write the options, not to tune a number.
- The head pose per second goes to `data/processed/pose/`. It is not committed.

## D26. Finding: R0 sees head turns, but not talk (2026-11-10)

This entry is a result, not a choice. All numbers come from `results/r0/`.

- ROC-AUC for cheat against no cheat, per second (`roc_auc_by_split.csv`):

  | Split part | Seconds | R0 | P1 (D23) |
  |---|---|---|---|
  | Protocol A, validation (3 actors) | 2733 | 0.837 | 0.501 |
  | Protocol A, test (9 real, 1 actor) | 11742 | 0.532 | 0.631 |
  | Protocol B, test of fold 1 | 5236 | 0.710 | 0.642 |
  | Protocol B, test of fold 2 | 6257 | 0.619 | 0.656 |
  | Protocol B, test of fold 3 | 6042 | 0.666 | 0.622 |
  | Protocol B, test of fold 4 | 6148 | 0.605 | 0.855 |

- All 24 subjects together: 0.628 (`summary.json`).
- Per subject (`per_subject.csv`): the ROC-AUC of the 15 actors goes from 0.594 to 0.904. The
  ROC-AUC of the 9 real exam takers goes from 0.493 to 0.664, which is near chance level.
- Per type code (`per_code.csv`): the suspicion is above 0.5 for 33.0% of the seconds of
  code 1 and 24.1% of code 6. It is above 0.5 for 10.2% of code 2, 7.3% of code 5, and 8.5%
  of the seconds with no cheat.
- Reading: the rule sees a head turn (codes 1 and 6). Talk (code 2) and a phone call (code
  5) need no head turn, so the rule does not see them. The real exam takers mostly talk
  (`results/eda/summary.md`). Thus R0 is weak on them.
- Face detection: MediaPipe finds a face in 87.5% of the seconds on average. The lowest
  subject has 60.1%.
- Comparison with P1 on protocol A: R0 is better on the validation actors, and P1 is better
  on the test set. Neither model is good on the real exam takers.
- Limits: these are scores per second. The event metrics and the alarm threshold come at
  milestone 6. The P1 values for protocol B come from the runs of D27.

## D27. Finding: P1 on protocol B (2026-11-11)

This entry is a result, not a choice. All numbers come from `results/p1_b_summary.csv`,
`results/p1_b_test_per_subject.csv`, and the 4 run folders `results/p1_b_f*`. The runs used
the code at `7fd330e` on a GPU NVIDIA L4, with the values of D21.

- Scores of the 4 folds:

  | Fold | Epochs (best) | Minutes | Test macro F1 | Baseline macro F1 | Test accuracy | Baseline accuracy | Test ROC-AUC | Validation ROC-AUC |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 13 (3) | 35.0 | 0.294 | 0.228 | 0.695 | 0.838 | 0.642 | 0.746 |
  | 2 | 19 (9) | 47.2 | 0.357 | 0.207 | 0.698 | 0.704 | 0.656 | 0.811 |
  | 3 | 15 (5) | 37.9 | 0.322 | 0.231 | 0.694 | 0.857 | 0.622 | 0.711 |
  | 4 | 11 (1) | 29.4 | 0.639 | 0.209 | 0.804 | 0.719 | 0.855 | 0.708 |
  | Mean | | 37.4 | 0.403 | 0.219 | 0.723 | 0.780 | 0.694 | 0.744 |
  | Std | | | 0.159 | | 0.054 | | 0.108 | 0.048 |

- Macro F1 is above the baseline in each fold. The accuracy is below the baseline in the
  folds 1, 2, and 3. The result changes much between folds: fold 4 is far above the others.
- The models overfit as on protocol A (D23). Early stopping kept an early epoch in each run.
- Per subject (`p1_b_test_per_subject.csv`, each subject is a test subject one time):
  - Real exam takers: the ROC-AUC of P1 goes from 0.594 to 0.909. P1 is better than R0
    for 9 of the 9 real exam takers.
  - Actors: the ROC-AUC of P1 goes from 0.295 to 0.949. P1 is better than R0 for
    8 of the 15 actors.
- Reading for research question 5: on protocol A, P1 trained on actors only, and its
  ROC-AUC on the real exam takers went from 0.367 to 0.871 (D23). On protocol B, the training
  folds hold real exam takers, and the values for them are higher. A model that learned from
  actors only does not transfer well to real exam takers.
- Reading for research question 1: P1 and R0 fail on different subjects. R0 is good for some
  actors where P1 is near or below chance level, for example the subjects 2 and 7. P1 is
  better for each real exam taker. No model is better for all subjects.
- Limits: one seed per fold. These are scores per second. The event metrics come at
  milestone 6. The models were not scored on their training subjects for protocol B.
- Tool: `python -m proctor_assist.evaluate --summary p1_b` writes the two summary files.

## D28. Score files per second stay out of the repository (decided by the owner in chat, 2026-11-11)

The owner chose this from one multiple-choice question in the chat (D16).

- Context: a score file (`scores_*.csv`, and `results/r0/scores.csv`) has the column
  `target`. It holds the class of D14 for each second of each subject. That is a derived
  form of the OEP labels. D13 and D22 say that label lists stay out of the public
  repository, because the license is not clear.
- Error by the AI: commit `9ae381c` added the three score files of the P1 run on protocol A
  to the public repository, against D13 and D22. The AI found this on 2026-11-11, before
  the next commit.
- Decision (owner): score files are not committed. `.gitignore` blocks them. They stay on
  the laptop in `results/` and on Drive in the run folders. The three committed files leave
  the repository with `git rm --cached`. The files on the laptop are not deleted.
- The metrics, the history per epoch, and the tables per subject and per type code stay in
  the repository.
- Cost: the metrics cannot be computed again from the repository alone. The three files are
  still in the git history.
- Options that the owner did not choose: keep the score files in the repository and correct
  D13 and D22. Remove the files from the git history with a force push.
- Follow-up: the question for the lecturer about the license also covers these files.

## D29. Milestone 5, round 1: P1s and the features of E1 (decided by the owner in chat, 2026-11-11)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record.

- New configuration P1s: the same MobileNetV2 as P1, with random initial weights, on
  protocols A and B. PLAN §5 gets one more row.
- E1 features: the backbone of the P1 run of the same split, frozen. Only the GRU is trained.
- Consequence: E1 has no augmentation, because the features are computed one time.
- Reasons, from the options that the owner chose:
  - P1s: B1 and P1 differ in the architecture and in the initial weights. P1s and P1 differ
    in the initial weights only. Thus P1s gives a clean answer to research question 2. The
    cost: about 3 to 4 hours of GPU time, estimated from the 173 minutes of the 5 P1 runs.
  - E1 on the P1 backbone: the only difference from P1 is one frame against one window, as
    in PLAN §5. The GRU trains in a short time. The cost: P1 learned the training subjects
    (D23). Its features are then too good on those subjects, and the GRU can fail on new
    people in the same way.
- Options that the owner did not choose: P1s on protocol B only, or no P1s. For E1, the
  ImageNet backbone with no fine-tuning, or a CNN and a GRU that train together.
- AI choices, easy to change (D10):
  - B1: 4 blocks of convolution (32, 64, 128, and 256 filters), batch normalization, ReLU,
    and max pooling. Then global average pooling and the output layer.
  - E1: the feature of a frame is the output of the last hidden dense layer of P1, with 128
    values. One GRU layer with 64 units, then the output layer.
  - B1, P1s, and E1 use the sampling, the epochs, the early stopping, and the optimizer
    values of P1 (D21). The config files of B1 and P1s come from the P1 files, and a test
    checks that they differ only in their variable.
- Code: `models.build` creates the model that a config names. Each model gets only the
  config values that it knows.
- Open for round 2: the design of E2 (landmarks), after a measurement of the MediaPipe speed
  for hands and body.

## D30. E1 code, and the state of the B1 and P1s runs (2026-11-11)

- State of the queue of D29: 3 of the 10 runs finished in the night of 2026-11-10: `b1_a`,
  `b1_b` fold 1, and `b1_b` fold 2. Their files are in `results/b1_*`. The run of fold 3
  started and wrote no epoch. The log shows no error. The cause is not known. A stopped
  Colab runtime is likely.
- Open: `b1_b` folds 3 and 4, and the 5 runs of P1s. The B1 results get their entry when all
  5 runs are in the repository. Until then, the B1 values for protocol B are TBD.
- Decision (owner, in chat): the AI writes the E1 code first. The rest of the queue runs
  later, when the owner can keep the Colab tab open.
- E1 code (AI choices inside D29, easy to change):
  - `proctor_assist.features` computes the features of all stored frames with the P1 model
    of the same protocol and fold. It stores them one time per P1 run.
  - The training script finds the P1 run folder by its name. It stops if it finds no run or
    more than one run.
  - A window has the 15 frames of 3 s (D18). Training and scoring use one window per second.
  - The configs of E1 keep the training values of P1, and a test checks this.
- Not done: E1 has not run on real features yet. The tests use made-up features. The script
  that scores the training subjects (D23) does not work for E1.
- Lesson for long Colab jobs: a job at night with no person at the laptop stopped one time.
  The guard on the laptop saw it and shut the laptop down, as the owner asked.

## D31. Order of milestone 5: E1 and E2 before the rest of B1 and P1s (decided by the owner in chat, 2026-11-11)

The owner chose each point from a multiple-choice question in the chat (D16). The proposal
came from a second AI session that the owner started.

- Order: E1 first, then the design and the landmark extraction of E2, then the open runs of
  B1 and P1s.
- P1s keeps its 5 runs (D29). They run last. The owner can review this if time or compute
  units run low.
- Reason, from the option that the owner chose: E1 and E2 are the experiments that can
  change the conclusion. E2 goes at the cause in D23, because landmarks do not carry the
  look of a person. B1 and P1s are controls for research question 2.
- Fact behind the proposal (`results/*/metrics.json`): the test ROC-AUC of B1 on folds 1 and
  2 of protocol B is 0.608 and 0.681. P1 has 0.642 and 0.656 on the same folds.
- Options that the owner did not choose: finish B1 and P1s first. P1s on protocol B only. No
  P1s.
- The design of the experiments does not change.

## D32. E2 design (decided by the owner in chat, 2026-11-11)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record.

- Landmarks: face, hands, and body, from MediaPipe, on the original video, at 5 fps.
- Model input: selected points, normalized to the body of the subject. The origin is the
  middle of the shoulders, and the scale is the shoulder width.
- The extraction stores all raw points. Thus the input can change with no new extraction.
- Reasons, from the options that the owner chose: hands and body show a phone at the ear
  and a turn of the body, which the face alone does not show. A small input that does not
  depend on the seat position is hard to learn by heart for each person.
- Options that the owner did not choose: face and body only, or face only. All raw points,
  or hand-made features only.
- AI choices, easy to change (D10):
  - Input per frame: head yaw, pitch, and roll. Eye and mouth points of the face. The upper
    body points. The points of each hand. One flag per part that says if it was found.
  - A part that is not found gives zeros.
  - Window of 3 s, GRU with 64 units, and the training values of P1, as for E1.
  - E2 trains on the laptop CPU if it is fast enough. TBD until measured.
- Model files in `data/models/mediapipe/` (not committed), from the MediaPipe model storage
  of Google:
  - `hand_landmarker.task`, 7819105 bytes, SHA-256
    `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
  - `pose_landmarker_full.task`, 9398198 bytes, SHA-256
    `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
- First measurement on the laptop (2 subjects, 3000 frames, not a result file): about 82 ms
  per frame for the three models together. The extraction of all frames then needs about
  3 hours.
- Not done yet: the extraction code, the E2 model, and the tests.

## D33. Finding: E1 is not better than P1 (2026-11-11)

This entry is a result, not a choice. All numbers come from the 5 run folders
`results/e1_*`, `results/e1_b_summary.csv`, and `results/e1_b_test_per_subject.csv`. The
runs used the code at `0c9b579` on a GPU NVIDIA L4. E1 has 37508 trainable parameters.

  | Run | E1 test ROC-AUC | P1 test ROC-AUC | E1 test macro F1 | P1 test macro F1 | E1 best epoch |
  |---|---|---|---|---|---|
  | Protocol A | 0.644 | 0.631 | 0.422 | 0.423 | 1 |
  | Protocol B fold 1 | 0.622 | 0.642 | 0.279 | 0.294 | 1 |
  | Protocol B fold 2 | 0.642 | 0.657 | 0.369 | 0.357 | 1 |
  | Protocol B fold 3 | 0.621 | 0.622 | 0.326 | 0.322 | 1 |
  | Protocol B fold 4 | 0.878 | 0.855 | 0.587 | 0.639 | 1 |
  | Protocol B mean (std) | 0.691 (0.125) | 0.694 (0.108) | 0.390 | 0.403 | |

- Per subject on protocol B, mean ROC-AUC: real exam takers 0.816 for E1 and
  0.803 for P1. Actors 0.658 for E1 and 0.665 for P1. E1 is better than
  P1 for 8 of the 9 real exam takers and for 9 of the 15 actors.
- Each E1 run kept the model of epoch 1. The validation loss rose after the first epoch.
- Reading: the GRU gets the frozen features of the P1 run of the same split. P1 learned its
  training subjects (D23), so the features of those subjects already separate the classes.
  The GRU learns them in one epoch and adds no transfer to new people. D29 named this cost
  before the runs.
- Reading for research question 3: with these features, a window of 3 s does not help.
  This is a result for this design of E1, not for temporal context in general.
- Feature time: about 2 minutes per P1 run on the L4. One E1 epoch took about 1.5 s.
- Limits: one seed per run. Scores per second. The event metrics come at milestone 6.

## D34. E2 code (2026-11-11)

These are AI choices inside D32. They are easy to change.

- `python -m proctor_assist.landmarks` reads every 5th frame of each video, so row k belongs
  to stored frame k (D18). It runs on the laptop, with 4 processes.
- Storage: `data/processed/landmarks/subjectN.npz`, never committed. It holds all raw points
  as float16: 478 face points, the face matrix, 33 body points with their visibility, and
  21 points for each of two hands. A part that is not found is NaN. The two hands are
  sorted from left to right in the image. Subject 1 needed 15 MB.
- Input per frame (`landmarks.frame_features`, 135 values):
  - Head yaw, pitch, and roll, divided by 90 degrees.
  - 13 face points: nose tip, eye corners and lids, inner lips, and mouth corners.
  - 9 body points: nose, ears, shoulders, elbows, and wrists.
  - The 21 points of each hand.
  - All points in pixels, minus the middle of the shoulders, divided by the shoulder width.
  - One flag each for face, body, first hand, and second hand.
  - A part that is not found gives zeros. Without a body, all points are zeros.
  - The z values of MediaPipe are not used.
- E2 uses the GRU class of E1 with 135 inputs (`models.MODELS["e2"]`). The configs of E2 equal
  those of E1 except the input, and a test checks this. E2 trains on the laptop CPU.
- Check of the point numbers on the 3648 frames of subject 1 with a face: the eyes are above
  the nose tip in 100.0% of the frames, and the nose tip is above the inner lips in 100.0%.
  The upper lid is above the lower lid in 99.6% and 98.1%. The two eyes and the two mouth
  corners are on different sides of the nose in 99.9% and 99.1%. The upper inner lip is at
  or above the lower inner lip in 65.7%, because the two lips touch when the mouth is
  closed. The body nose is above the shoulders in 93.7% of the frames.
- Speed on the laptop: subject 1 (4321 frames) took 6 min 44 s in one process.
- Full run on 2026-11-11: all 24 subjects in about 44 minutes with 4 processes, 414 MB in
  total. The run used the code of commit `24d076d` before it was committed, so
  `results/landmarks/run.json` says `git_dirty: true`. The frame count of each subject equals `results/frames/summary.csv`. Mean detection
  rates (`results/landmarks/summary.csv`):

  | Group | Face | Body | First hand | Second hand |
  |---|---|---|---|---|
  | Actors | 91.8% | 99.3% | 8.9% | 0.1% |
  | Real exam takers | 80.1% | 94.3% | 20.9% | 0.7% |

- A second hand is almost never found. The hand points are mostly zeros.
- E2 on the laptop CPU: one pass over the 9208 training windows of protocol A took 3.7 s.
  Thus E2 trains on the laptop, as D32 allows.
- No fitted statistic: the normalization uses the shoulders of the same frame. Thus there
  is no risk of test data in a statistic.

## D35. Finding: E2 is below P1 on new people (2026-11-11)

This entry is a result, not a choice. All numbers come from the 5 run folders `results/e2_*`,
`results/e2_b_summary.csv`, and `results/e2_b_test_per_subject.csv`. The runs used the code at
`24dcdc0` on the laptop CPU. E2 has 38852 trainable parameters. One run took from
50 s to 101 s.

  | Run | E2 test ROC-AUC | P1 test ROC-AUC | E2 test macro F1 | Baseline macro F1 | E2 epochs (best) |
  |---|---|---|---|---|---|
  | Protocol A | 0.569 | 0.631 | 0.276 | 0.210 | 20 (13) |
  | Protocol B fold 1 | 0.573 | 0.642 | 0.457 | 0.228 | 20 (16) |
  | Protocol B fold 2 | 0.589 | 0.657 | 0.351 | 0.207 | 11 (1) |
  | Protocol B fold 3 | 0.619 | 0.622 | 0.302 | 0.231 | 20 (10) |
  | Protocol B fold 4 | 0.669 | 0.855 | 0.366 | 0.209 | 20 (15) |
  | Protocol B mean (std) | 0.612 (0.042) | 0.694 (0.108) | 0.369 | | |

- Mean ROC-AUC per subject on protocol B:

  | Group | E2 | E1 | P1 | R0 |
  |---|---|---|---|---|
  | Actors | 0.661 | 0.658 | 0.665 | 0.753 |
  | Real exam takers | 0.601 | 0.816 | 0.803 | 0.550 |

- E2 is better than P1 for 5 of the 15 actors and for 0 of the 9 real exam takers.
- E2 overfits less than P1 and E1: most runs kept a late epoch, and the spread between the
  folds is smaller. But it separates cheating from no cheat less well.
- Reading: the hypothesis of D31 was that landmarks transfer better to new people, because
  they do not carry the look of a person. The data does not support it for this design. The
  real exam takers mostly talk (code 2). Talk shows in the mouth and in a second person in
  the image. The landmarks of E2 see the mouth of the subject, but not the second person.
  This reading is not tested.
- Limits: one seed per run. One design of the input (D34). Scores per second. The event
  metrics come at milestone 6.

## D36. Finding: B1 and P1s, the controls for research question 2 (2026-11-11)

This entry is a result, not a choice. All numbers come from the run folders `results/b1_*`
and `results/p1s_*`, `results/b1_b_summary.csv`, `results/p1s_b_summary.csv`, and the two
`*_test_per_subject.csv` files. All runs used a GPU NVIDIA L4. B1 folds 1 and 2 and
protocol A used the code at `de3fbb1`, and the other runs used `0c9b579`. Between these
commits, only the E1 path changed. The frame path of B1 and P1s is the same. B1 has 390404
trainable parameters. P1 and P1s have 3493864.

  | Run | P1 test ROC-AUC | P1s test ROC-AUC | B1 test ROC-AUC | P1 / P1s / B1 test macro F1 | Baseline macro F1 |
  |---|---|---|---|---|---|
  | Protocol A | 0.631 | 0.600 | 0.544 | 0.423 / 0.288 / 0.239 | 0.210 |
  | Protocol B fold 1 | 0.642 | 0.587 | 0.608 | 0.294 / 0.275 / 0.384 | 0.228 |
  | Protocol B fold 2 | 0.657 | 0.670 | 0.681 | 0.357 / 0.394 / 0.352 | 0.207 |
  | Protocol B fold 3 | 0.622 | 0.518 | 0.592 | 0.322 / 0.278 / 0.314 | 0.231 |
  | Protocol B fold 4 | 0.855 | 0.656 | 0.669 | 0.639 / 0.323 / 0.363 | 0.209 |
  | Protocol B mean (std) | 0.694 (0.108) | 0.608 (0.070) | 0.638 (0.044) | 0.403 / 0.317 / 0.353 | |

- Mean ROC-AUC per subject on protocol B:

  | Group | P1 | P1s | B1 | R0 |
  |---|---|---|---|---|
  | Actors | 0.665 | 0.620 | 0.710 | 0.753 |
  | Real exam takers | 0.803 | 0.654 | 0.740 | 0.550 |

- Compared with P1 per subject: P1s is better for 5 of the 15 actors and 2 of the 9 real
  exam takers. B1 is better for 9 of the 15 actors and 4 of the 9 real exam takers.
- Epochs on protocol B (mean): P1s ran 14.75 and kept epoch 4.75. B1 ran 15.0 and kept
  epoch 6.25. One run took from 21 to 52 minutes.
- Reading for research question 2: P1 (ImageNet weights) is better than P1s (random
  weights) on 4 of the 5 splits and on the mean of protocol B. Fold 2 is the exception.
  Thus pretraining helps here, mostly on the real exam takers.
- Reading for B1: on the mean of protocol B, the small CNN from scratch is better than the
  large network from scratch (P1s). B1 also has the smallest spread of the three. It is
  below P1 on the mean and on the real exam takers, but above P1 on the actors. On
  protocol A, B1 is the lowest of the three.
- Limits: one seed per run, so a difference of a few hundredths between two runs can be
  noise. Scores per second. The event metrics come at milestone 6.
- Not in the repository: the Drive folder `b1_b_f3_20261110-203400_de3fbb1` holds only a
  config and an env file of the stopped run of D30. It stays until the owner decides.

## D37. Milestone 6: event metric, threshold, and changed conditions (decided by the owner in chat, 2026-11-11)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record. The answers, word for
word (the chat was in Indonesian):

1. E3: "Tunda (Recommended)". Milestones 6 to 8 come first. E3 runs only if time is left.
2. Event: "Rangkaian detik cheat (Recommended)". One event is a run of consecutive seconds
   with any cheat class.
3. Tolerance: "2 s (Recommended)", at the start and at the end of an event.
4. Threshold: "≤2 FA/jam per peserta (Recommended)". The threshold is chosen on validation
   subjects only, for at most 2 false alarms per hour per exam taker. The full curve of
   detected events against false alarms per hour is reported for all models.
5. Changed conditions, levels: "2 tingkat: sedang, berat (Recommended)". 4 types (low
   light, blur, compression, lower frame rate) at 2 levels (moderate, strong).
6. Changed conditions, models: "P1, B1, E2, R0 (Recommended)".
7. Changed conditions, subjects: "Protokol B, 4 fold (Recommended)". The threshold comes from
   the validation subjects without changes.

- Reasons, from the options that the owner chose:
  - E3 is optional, and the event metric and the app are required deliverables.
  - The app score does not separate the classes (D10), so an event ignores the class.
  - A window of 3 s that ends up to 2 s after an event still holds frames of the event, and
    the label times have a resolution of 1 s.
  - With 9 tiles on the dashboard, 2 false alarms per hour per exam taker give about 18 per
    hour for the proctor.
- Facts behind the options (from the labels after D14, not a result file of a model): 442
  events, median duration 6 s, and 25% of the events last 2 s or less. The gap between two
  events is 2 s or less for 20% of the gaps, and 5 s or less for 44%. All 24 subjects have
  about 5.1 hours with no cheat, so one validation fold of protocol B has about 1.3 hours.
- Cost that the owner accepted: with about 1.3 hours of validation per fold, 2 false alarms
  per hour means 2 or 3 alarms. The threshold is noisy.
- Options that the owner did not choose: E3 now, or no E3. One label row per event, or one
  run per class. A tolerance of 0 s or 5 s. A budget of 6 per hour, two budgets, or the
  best event F1. 1 or 3 levels. All 6 models, or P1 and E2 only. Protocol A only.

## D38. False alarm count, and the event code (2026-11-11)

Why this entry exists: the first version of the event code (not committed) had a flaw. One
alarm over the whole exam detected every event and counted as no false alarm, because it
touched an event. Thus on folds 1, 2, and 4 of protocol B, the chosen threshold fell to about
0, and most models "detected" 100% of the events. The AI saw this in the first results and
stopped. These results are not kept.

Decided by the owner in chat (D16), 2026-11-11, with a final recap. The answer, word for word:
"Bagian di luar event (Recommended)".

- Each part of an alarm outside all events, widened by 2 s at each end, is one false alarm.
- An event is detected if one second of the event, widened by 2 s at each end, is an alarm
  second.
- Event, tolerance, budget, and the other points of D37 do not change.
- Reason, from the option that the owner chose: no new parameter, and a long alarm costs the
  proctor a long review in the review list of the app. An alarm over the whole exam now gives
  48.6 false alarms per hour (all 24 subjects, R0 seconds), far above the budget.
- Cost that the owner accepted: a correct alarm that goes on for more than 2 s after the event
  also counts one false alarm.
- Options that the owner did not choose: an alarm again every R seconds, or alarm parts of N
  seconds.

AI choices in `proctor_assist.events`, easy to change:
- An alarm second is a scored second with a suspicion at or above the threshold. There is no
  smoothing. A masked second (D14) ends an event and an alarm.
- False alarms per hour = false alarms / all scored seconds of the part, in hours. This is the
  load per exam taker.
- Threshold candidates: 1001 quantiles of the validation suspicion, plus infinity (no alarm).
  The choice is the most detected events at no more than 2 false alarms per hour on
  validation. Ties go to the higher threshold.
- The delay of a detected event: its first alarm second inside the widened event, minus the
  event start. It can be negative, down to -2 s.
- R0 gets a threshold per fold from the validation subjects of that fold, on the same seconds
  as the trained models.
- Protocol A is also evaluated, with its 3 validation subjects.
- `results/events/curves.csv` holds the curve of each run on validation and test, without the
  points that repeat the previous point.

## D39. Finding: at 2 false alarms per hour, the models detect few events (2026-11-11)

This entry is a result, not a choice. All numbers come from `results/events/summary.csv`,
`results/events/protocol_b.csv`, and `results/events/curves.csv`. They were computed on the
laptop CPU from the saved scores of the runs in D33, D35, and D36, and from `results/r0/`.
Protocol B has 441 test events in 6.58 hours of scored test video.

- Protocol B, threshold from validation (mean over 4 folds, std in brackets):

  | Model | Test events detected | Test false alarms per hour | Real exam takers: detected / false alarms per hour | Actors: detected / false alarms per hour |
  |---|---|---|---|---|
  | R0 | 0.000 (0.000) | 0.00 (0.00) | 0.000 / 0.00 | 0.000 / 0.00 |
  | B1 | 0.198 (0.144) | 18.08 (13.91) | 0.162 / 6.96 | 0.243 / 27.02 |
  | P1 | 0.099 (0.049) | 4.85 (5.11) | 0.127 / 7.68 | 0.096 / 3.79 |
  | P1s | 0.074 (0.081) | 23.42 (31.55) | 0.082 / 9.39 | 0.054 / 29.29 |
  | E1 | 0.151 (0.121) | 4.45 (2.48) | 0.182 / 6.72 | 0.133 / 3.78 |
  | E2 | 0.016 (0.020) | 1.12 (1.36) | 0.017 / 1.92 | 0.014 / 0.52 |

- The threshold does not transfer well from validation to test. Validation allows at most 2
  false alarms per hour, but the test mean is above 2 for B1, P1, P1s, and E1. P1s on fold 3
  has 66.7 per hour on test.
- R0 gets the threshold "no alarm" in every fold. Its score is 1 for 0.7% of the seconds
  (a head turn of 30 degrees or more, D25). These seconds alone give more than 2 false alarms
  per hour on validation.
- E2 gets the threshold "no alarm" on fold 4, and detects no test event on folds 3 and 4.
- View with no threshold, from the test curves. This is for comparison only, not a threshold
  choice. The value is the largest share of events detected, mean over the 4 folds:

  | Limit on test false alarms per hour | E1 | P1 | R0 | E2 | P1s | B1 |
  |---|---|---|---|---|---|---|
  | 2 | 0.089 | 0.081 | 0.000 | 0.015 | 0.041 | 0.017 |
  | 10 | 0.270 | 0.223 | 0.163 | 0.140 | 0.093 | 0.068 |
- Reading: at the budget of D37, no model detects more than about a fifth of the events, and
  E1 and P1 give the best balance. With about 1.6 hours of scored validation video per fold,
  2 false alarms per hour means 3 alarms or less, so the threshold is noisy (D37 named this
  cost).
- Limits: one seed per run. The event metric counts each event once, whatever its length.

## D40. Code of the tests under changed conditions (2026-11-11)

These are AI choices inside D37. They are easy to change. Code: `proctor_assist.conditions`.

- Values of the two levels (moderate, strong):

  | Kind | Change | Moderate | Strong |
  |---|---|---|---|
  | Low light | factor on the brightness of each pixel | 0.5 | 0.25 |
  | Blur | Gaussian blur, sigma in pixels of the 640 x 480 frame | 8 | 16 |
  | Compression | JPEG quality | 15 | 7 |
  | Lower frame rate | frames per second of the camera | 2.5 | 1 |

- Rule for blur and compression: moderate is severity 3 and strong is severity 5 of the
  benchmark ImageNet-C (Hendrycks and Dietterich, 2019). Checked in the code of the authors,
  `ImageNet-C/create_c/make_imagenet_c.py` in the GitHub repository hendrycks/robustness: JPEG
  quality 25, 18, 15, 10, 7, and blur sigma 1, 2, 3, 4, 6 for severities 1 to 5, on images of
  224 x 224.
- The resize from 640 x 480 to 224 x 224 divides the width by 2.86 and the height by 2.14.
  Thus a sigma of 8 and 16 at 640 x 480 is about 3 and 6 at 224 x 224 (2.8 and 5.6 in width,
  3.7 and 7.5 in height).
- A first choice of the AI (sigma 3 and 6, JPEG quality 20 and 7) was too weak for blur: about
  severity 1 and 2. The AI changed it before any blur or JPEG frame was extracted.
- Low light: ImageNet-C has no darker level, so the factor is half and a quarter of the light.
- Lower frame rate: the video has 25 fps, and the models read 5 fps (D18). Thus 2.5 and 1 fps
  are half and a fifth of the input rate.
- Low light has no added sensor noise. A real dark camera adds noise, so this test is milder
  than a real dark room.
- An image condition changes each video frame before every other step, as in a worse camera.
  Then the frame goes to MediaPipe (E2, R0) and to the resize and the JPEG step of D18
  (P1, B1). Thus each model gets the frame through its own normal path.
- A lower frame rate shows each stored frame for longer: at 2.5 fps, stored frame k shows frame
  2 x floor(k / 2), and at 1 fps, frame 5 x floor(k / 5). It needs no new extraction. P1, B1,
  and R0 read one frame per second, so for them only the age of that frame changes.
- Storage: per condition and subject, only the E2 input (135 values per frame), the face
  matrix (for R0), and the scoring frame of each second (for P1 and B1). The raw points are not
  stored, because drive D has 6.4 GB free. Folder: `data/processed/conditions/`, never committed.
- R0 reads the face matrix of the E2 extraction. Check on subjects 1, 3, and 9 without a
  change: the R0 scores from this matrix equal `results/r0/scores.csv` (largest difference 0.0).
- Scores on the laptop CPU. The condition "none" scores the data without a change with the same
  code. Check on the first 600 test rows of P1 fold 1: the CPU scores differ from the saved GPU
  scores by at most 0.0021 in P(no cheat). `results/conditions/run.json` holds this difference
  for all models and folds.
- The extraction runs with 6 processes. Measured on the laptop: P1 needs about 17 ms and B1
  about 16 ms per frame in a batch of 256 (8 threads). These are single measurements, not the
  compute results of milestone 7.

## D41. Finding: blur hurts all models, and low light breaks B1 (2026-11-12)

This entry is a result, not a choice. All numbers come from `results/conditions/summary.csv`
and `results/conditions/protocol_b.csv`. They were computed on the laptop CPU with the code of
D40 (git 3744ef3, see `results/conditions/run.json`). The test subjects of the 4 folds of
protocol B have 441 events in 6.58 hours. Each model keeps the threshold that it got from its
validation subjects without a change (D37).

- Check of the condition "none": it gives the same detected share and false alarms per hour as
  D39 for all 4 models. Against the saved scores, the largest difference in P(no cheat) is
  0.0087 for P1 and 0.0020 for B1 (CPU against GPU). For R0 and E2 it is 0.000005, which is the
  rounding to 5 decimals of the score files.
- ROC-AUC for "cheat" per second (mean over the 4 folds, std in brackets):

  | Condition | R0 | B1 | P1 | E2 |
  |---|---|---|---|---|
  | none | 0.650 (0.048) | 0.638 (0.044) | 0.694 (0.108) | 0.612 (0.042) |
  | low_light_moderate | 0.645 (0.052) | 0.527 (0.077) | 0.719 (0.115) | 0.613 (0.038) |
  | low_light_strong | 0.640 (0.048) | 0.450 (0.057) | 0.632 (0.066) | 0.621 (0.027) |
  | blur_moderate | 0.588 (0.027) | 0.520 (0.072) | 0.591 (0.112) | 0.545 (0.022) |
  | blur_strong | 0.497 (0.020) | 0.481 (0.044) | 0.559 (0.085) | 0.495 (0.010) |
  | jpeg_moderate | 0.638 (0.053) | 0.633 (0.057) | 0.685 (0.099) | 0.615 (0.038) |
  | jpeg_strong | 0.625 (0.072) | 0.638 (0.066) | 0.684 (0.066) | 0.546 (0.062) |
  | fps_moderate | 0.651 (0.052) | 0.637 (0.044) | 0.694 (0.109) | 0.612 (0.042) |
  | fps_strong | 0.656 (0.055) | 0.635 (0.047) | 0.691 (0.107) | 0.605 (0.037) |

- Test events detected / test false alarms per hour (mean over the 4 folds), with the threshold
  from validation without a change:

  | Condition | R0 | B1 | P1 | E2 |
  |---|---|---|---|---|
  | none | 0.000 / 0.00 | 0.198 / 18.08 | 0.099 / 4.85 | 0.016 / 1.12 |
  | low_light_moderate | 0.000 / 0.00 | 0.671 / 56.30 | 0.046 / 2.36 | 0.016 / 0.80 |
  | low_light_strong | 0.000 / 0.00 | 0.790 / 37.85 | 0.086 / 18.23 | 0.021 / 0.52 |
  | blur_moderate | 0.000 / 0.00 | 0.414 / 41.99 | 0.050 / 7.22 | 0.005 / 1.20 |
  | blur_strong | 0.000 / 0.00 | 0.407 / 42.40 | 0.000 / 0.00 | 0.003 / 1.38 |
  | jpeg_moderate | 0.000 / 0.00 | 0.202 / 26.94 | 0.089 / 4.45 | 0.009 / 0.32 |
  | jpeg_strong | 0.000 / 0.00 | 0.356 / 36.61 | 0.076 / 6.71 | 0.007 / 0.98 |
  | fps_moderate | 0.000 / 0.00 | 0.189 / 17.36 | 0.092 / 4.20 | 0.019 / 1.12 |
  | fps_strong | 0.000 / 0.00 | 0.187 / 19.70 | 0.089 / 4.36 | 0.024 / 1.12 |

- Blur is the hardest condition. At strong blur, the ROC-AUC of R0, B1, and E2 is about 0.5,
  the level of chance. P1 keeps 0.559, but its scores stay below the threshold, so it gives no
  alarm in any fold. R0 and E2 read MediaPipe landmarks. Blur probably makes the landmarks less
  exact or makes the face detection fail. Not verified: the share of frames with a face under
  each condition is not measured yet.
- Low light breaks B1. Its ROC-AUC drops to 0.527 and 0.450 (below chance), and it gives 56.30
  and 37.85 false alarms per hour. Its high detected share comes from alarms in most seconds,
  not from better detection. At strong low light, B1 detects all events on folds 1, 3, and 4,
  with 40.6 to 51.8 false alarms per hour.
- P1 keeps its ROC-AUC at moderate low light (0.719) and drops to 0.632 at strong low light.
  There its false alarms rise to 18.23 per hour. R0 and E2 change little under low light.
- JPEG compression: P1 and B1 keep their ROC-AUC, but the false alarms of B1 rise to 26.94 and
  36.61 per hour. E2 drops to 0.546 at quality 7.
- Lower frame rate: the ROC-AUC of each model changes by at most 0.007. This is expected,
  because the target is per second, and P1, B1, and R0 read one frame per second.
- P1 has the highest ROC-AUC of the 4 models under 8 of the 9 conditions. Under strong low
  light, R0 is higher (0.640 against 0.632).
- Reading: a changed camera shifts the scores. Thus the threshold from clean validation is the
  weak point, and the false alarms per hour change more than the ROC-AUC. Example: under strong
  JPEG, B1 keeps a ROC-AUC of 0.638, but its false alarms go from 18.08 to 36.61 per hour.
- A possible step for the app (milestone 8, not decided): a warning to the proctor when the
  video is dark or blurred.
- Limits: one seed per run and 4 folds, with a large std (about 0.1 for P1). The image changes
  are simulated (D40), and low light has no sensor noise. R0 detects no event under any
  condition, because its threshold is "no alarm" (D39).

## D42. Milestone 7: capacity test, deployment model, and error analysis (decided by the owner in chat, 2026-11-12)

The owner chose each point from a multiple-choice question in the chat, and then accepted a
final recap (D16). The AI wrote the options and typed this record. The answers, word for
word (the chat was in Indonesian):

1. Capacity: "Load test N stream (Recommended)". N videos run at the same time through the
   real pipeline of the app, video decoding included, on the laptop. N is the largest number
   that keeps up with 1 s of video per second. The time of each step is also recorded. The
   models are R0, B1, P1, E1, and E2. P1s has the architecture of P1, so it is not measured.
2. Deployment model: "Angka validation (Recommended)". From the models that run in real time,
   the app takes the one that detects the most events on the validation subjects of protocol
   B, at the threshold of D37. The test numbers are reported, but they are not used for the
   choice. If no new model beats P1, the app uses P1 (PLAN §7).
3. Error analysis: "Per kelas & tipe OEP (Rec.)", "Aktor vs peserta asli (Rec.)", "Panjang
   event & wajah (Rec.)", and "Lihat frame false alarm". These are: detection per class of
   D14 and per OEP code with the confusion matrix, actors against real exam takers and per
   subject, detection by event length, the share of frames with a face under each condition,
   and a look at the frames of the 20 largest false alarms of P1 and E1.

- Reasons, from the options that the owner chose:
  - A load test counts the contention for CPU and memory between streams. A division of
    1 s by the cost of one stream does not, and it gives a number that is too high.
  - The choice on validation keeps the test numbers honest. The cost that the owner accepted:
    validation has about 1.6 hours of video per fold, so the numbers are noisy.
- Facts behind the options (`results/events/summary.csv`): on the validation subjects of
  protocol B, the mean share of events detected is 0.139 for E1, 0.098 for P1, 0.050 for P1s,
  0.042 for E2, 0.036 for B1, and 0 for R0. Thus E1 wins if it runs in real time.
- Consequence for the app (in the recap that the owner accepted): the app uses the protocol A
  copy of the chosen architecture. The app shows the 9 real exam takers of the test set of
  protocol A, and only this copy did not see them in training.
- The descriptions of the false alarm frames come from the AI. The owner checks them. The
  frames are not committed.
- Options that the owner did not choose: the cost of one stream divided into 1 s. A choice
  with an extra rule on the changed conditions, or a choice on the test numbers of protocol B.
