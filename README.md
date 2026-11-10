# Proctor Assist: Cheating Detection for Online Exam Proctors

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). The system watches the
webcam video of online exam takers and flags moments for a proctor to review. The proctor
decides. The system never labels a person as a cheater. Goals, scope, and constraints are in
[docs/PLAN.md](docs/PLAN.md). Design decisions are in [docs/decisions.md](docs/decisions.md).

Results: TBD. No experiment has run yet.

## Repository layout

- `src/proctor_assist/`: Python package with all core logic.
- `tests/`: pytest tests.
- `manifests/`: SHA-256 manifest of the raw data (`docs/decisions.md` D11).
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

1. Open `notebooks/colab.ipynb` in Colab from the GitHub repository.
2. Select a GPU runtime.
3. Run the cells in order.
