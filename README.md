# Proctor Assist: Cheating Detection for Online Exam Proctors

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). The system watches the
webcam video of online exam takers and flags moments for a proctor to review. The proctor
decides. The system never labels a person as a cheater. Goals, scope, and constraints are in
[docs/PLAN.md](docs/PLAN.md). Design decisions are in [docs/decisions.md](docs/decisions.md).

Results: TBD. No experiment has run yet.

## Repository layout

- `src/proctor_assist/`: Python package with all core logic.
- `tests/`: pytest tests.
- `notebooks/colab.ipynb`: Colab bootstrap. It mounts Drive, clones or pulls, installs, and runs commands.
- `docs/`: plan, decisions, and AI usage log.

Data, checkpoints, and run folders are not in this repository. Locally, they live in the
gitignored `data/` and `runs/` folders. In Colab, they live on Google Drive under
`MyDrive/proctor-assist/` (`docs/decisions.md` D3 and D8).

## Data

The project uses the MSU Online Exam Proctoring (OEP) dataset (Atoum et al., IEEE
Transactions on Multimedia, 2017), through the Kaggle mirror `raajanwankhade/oep-dataset`.
The download script is part of milestone 1. Never commit videos, frames, or landmark files.

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
