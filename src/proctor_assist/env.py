"""Storage paths and environment records. Every run folder gets an env.json."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Storage for data/ and runs/: Google Drive in Colab (GPU jobs, D8), else the repo folder
# (both gitignored). Set PROCTOR_ROOT to use another folder, e.g. a Drive for Desktop folder.
_COLAB = Path("/content/drive/MyDrive/proctor-assist")
ROOT = Path(os.environ.get("PROCTOR_ROOT") or (_COLAB if _COLAB.parent.exists() else REPO))
# Folder for data that a job reads many times, such as the frames. In Colab, the Drive folder is
# slow for many small files, so this is the local disk of the runtime. A runtime reset empties it,
# and `python -m proctor_assist.pack unpack` fills it again (D20).
FAST_DATA = Path("/content/proctor-data") if ROOT == _COLAB else ROOT / "data"


def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def env_info():
    status = _run(["git", "status", "--porcelain"])
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
        # True means the run used uncommitted code, so the sha alone does not reproduce it.
        "git_dirty": bool(status) if status is not None else None,
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }


def new_run_dir(name):
    """Create ROOT/runs/<name>_<UTC time>_<git sha> and write env.json into it.

    Fails if the folder exists, so a run never overwrites another run.
    """
    info = env_info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{name}_{stamp}_{info['git_sha'] or 'nogit'}"
    run_dir.mkdir(parents=True)
    (run_dir / "env.json").write_text(json.dumps(info, indent=2))
    return run_dir


if __name__ == "__main__":
    # Setup check: write env.json to a new run folder under ROOT.
    run_dir = new_run_dir("setup-check")
    info = json.loads((run_dir / "env.json").read_text())
    print(f"run folder: {run_dir}")
    for key in ("git_sha", "git_dirty", "gpu", "python"):
        print(f"{key}: {info[key]}")
