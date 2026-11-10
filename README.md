# Proctor Assist: Cheating Detection for Online Exam Proctors

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). The system watches the
webcam video of online exam takers and flags moments for a proctor to review. The proctor
decides. The system never labels a person as a cheater. Goals, scope, and constraints are in
[docs/PLAN.md](docs/PLAN.md). Design decisions are in [docs/decisions.md](docs/decisions.md).

Results so far: the first P1 run on protocol A is in `results/p1_a_20261110-132143_edfc724/`.
Its reading is in `docs/decisions.md` (D23). All other results are TBD.

## Repository layout

- `src/proctor_assist/`: Python package with all core logic.
- `tests/`: pytest tests.
- `manifests/`: SHA-256 manifest of the raw data (`docs/decisions.md` D11).
- `splits/subjects.json`: the fixed subject splits for protocols A and B (`docs/decisions.md` D15).
- `results/`: small result files that scripts write. `results/eda/` holds the EDA tables and figures.
- `notebooks/colab.ipynb`: Colab bootstrap. It mounts Drive, clones or pulls, installs, and runs commands.
- `docs/`: plan, decisions, and AI usage log.

Data, checkpoints, and run folders are not in this repository. Locally, they live in the
gitignored `data/` and `runs/` folders. In Colab, they live on Google Drive under
`MyDrive/proctor-assist/` (`docs/decisions.md` D3 and D8).

## Data

The project uses the MSU Online Exam Proctoring (OEP) dataset (Atoum et al., IEEE
Transactions on Multimedia, 2017), through the Kaggle mirror `raajanwankhade/oep-dataset`.
The download script gets only the files that the project uses: `READ_ME.txt`, the 24 `gt.txt`
files, and the 24 webcam videos (`docs/decisions.md` D11). Never commit videos, frames, or
landmark files.

To download the data:

1. Put your Kaggle API token in `~/.kaggle/access_token`, or set `KAGGLE_API_TOKEN`.
2. Run the download script. It saves the files to `data/raw/oep/`. If `PROCTOR_ROOT` is set,
   it uses `$PROCTOR_ROOT/data/raw/oep/` instead.

```bash
.venv/Scripts/python -m proctor_assist.download
```

The first run writes `manifests/oep_raw.sha256`. Each later run checks every file against it.
If you run the script again after an interruption, it skips the complete files.

## Run the EDA

The EDA reads the labels and the webcam videos in `data/raw/oep/`. It writes tables,
`summary.md`, and figures to `results/eda/`. It writes sample frames to `data/eda/`, which is
never committed.

```bash
.venv/Scripts/python -m proctor_assist.eda
```

## Prepare the data for training

Run these steps in order. The subject splits come first, so that no frame is extracted
before each subject has its split (`docs/decisions.md` D15 and D18).

1. Write the subject splits to `splits/subjects.json`. The file is committed, so this step is
   necessary only after a change to the split design.

```bash
.venv/Scripts/python -m proctor_assist.splits
```

2. Extract the frames at 5 fps and 224 x 224 to `data/processed/frames/`. If the run stops,
   start it again. It skips the subjects that are complete.

```bash
.venv/Scripts/python -m proctor_assist.frames
```

3. Count the windows per class for each split, in `results/windows/summary.csv`.

```bash
.venv/Scripts/python -m proctor_assist.windows
```

## Run locally

1. Create the virtual environment and install the package.
2. Run the setup check. It writes `env.json` to a new folder in `runs/`.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

```bash
.venv/Scripts/python -m proctor_assist.env
```

On Linux or macOS, use `.venv/bin/python` instead of `.venv/Scripts/python`.

## Run unit tests locally

```bash
.venv/Scripts/python -m pytest
```

## Run GPU jobs in Colab

The frames go to Colab as one zip file per subject, through Google Drive
(`docs/decisions.md` D20). Do this step one time on the laptop, after the frame extraction.
Drive for Desktop then uploads the files.

```bash
.venv/Scripts/python -m proctor_assist.pack pack "G:/My Drive/proctor-assist"
```

To train a model, give its config file to the training script. The run folder gets the
config, the history per epoch, the checkpoints, and the metrics (`docs/decisions.md` D22).
Add `--smoke` for a short check of the pipeline on CPU.

```bash
python -m proctor_assist.train configs/p1_a.yaml
```

1. Open `notebooks/colab.ipynb` in Colab from the GitHub repository.
2. Select a GPU runtime.
3. Run the cells in order. The cell with `pack unpack` checks the zip files against
   `manifests/frames_zip.sha256` and opens them on the local disk of the runtime.
