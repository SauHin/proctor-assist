"""Move the stored frames from the laptop to Colab as one zip file per subject (D20).

On the laptop:  python -m proctor_assist.pack pack "G:/My Drive/proctor-assist"
In Colab:       python -m proctor_assist.pack unpack

`pack` writes the zip files into the Google Drive folder and writes manifests/frames_zip.sha256.
It also copies the gt.txt files, because the training builds its targets from them.
`unpack` checks each zip file and each gt.txt against its manifest, then opens the zip files on
the fast local disk. Thus Colab trains on the same bytes that the laptop has.
"""
import sys
import zipfile
from pathlib import Path

from proctor_assist import frames
from proctor_assist.download import MANIFEST as RAW_MANIFEST
from proctor_assist.download import read_manifest, sha256, write_manifest
from proctor_assist.env import REPO, ROOT
from proctor_assist.labels import OEP, SUBJECTS

MANIFEST = REPO / "manifests" / "frames_zip.sha256"
ZIP_DIR = Path("data") / "processed" / "frames_zip"  # below the Drive folder of the project


def pack(drive_root, subjects=SUBJECTS, manifest=MANIFEST):
    """Write subjectN.zip for each subject into the Drive folder, and the manifest of their hashes."""
    out = Path(drive_root) / ZIP_DIR
    out.mkdir(parents=True, exist_ok=True)
    known = read_manifest(manifest) if manifest.exists() else {}
    hashes = {}
    for s in subjects:
        files = sorted((frames.OUT / f"subject{s}").iterdir())
        if not any(f.name == "_done.json" for f in files):
            raise SystemExit(f"subject{s} is not complete. Run proctor_assist.frames first.")
        path = out / f"subject{s}.zip"
        if path.exists() and sha256(path) == known.get(path.name):  # correct file from an earlier run
            hashes[path.name] = known[path.name]
            continue
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:  # JPEG is compressed already
            for f in files:
                # A fixed date gives the same zip bytes on each run.
                z.writestr(zipfile.ZipInfo(f.name, date_time=(1980, 1, 1, 0, 0, 0)), f.read_bytes())
        hashes[path.name] = sha256(path)
        print(f"{path.name}: {len(files)} files", flush=True)
    write_manifest(manifest, hashes)
    return hashes


def pack_labels(drive_root):
    """Copy the 24 gt.txt files into the Drive folder. The training in Colab builds its targets from them."""
    for s in SUBJECTS:
        target = Path(drive_root) / "data" / "raw" / "oep" / f"subject{s}" / "gt.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((OEP / f"subject{s}" / "gt.txt").read_bytes())


def check_labels(drive_root=ROOT):
    """Fail if a gt.txt in the Drive folder differs from the manifest of the raw data (D11)."""
    expected = read_manifest(RAW_MANIFEST)
    for s in SUBJECTS:
        rel = f"subject{s}/gt.txt"
        if sha256(Path(drive_root) / "data" / "raw" / "oep" / rel) != expected[rel]:
            raise SystemExit(f"{rel} in {drive_root} differs from the manifest of the raw data.")


def unpack(drive_root=ROOT, subjects=SUBJECTS, manifest=MANIFEST):
    """Check each zip file against the manifest, then open it into the frame folder."""
    expected = read_manifest(manifest)
    for s in subjects:
        target = frames.OUT / f"subject{s}"
        if (target / "_done.json").exists():
            continue
        path = Path(drive_root) / ZIP_DIR / f"subject{s}.zip"
        if sha256(path) != expected[path.name]:
            raise SystemExit(f"{path} differs from the manifest. The upload is not complete, or the file changed.")
        with zipfile.ZipFile(path) as z:
            z.extractall(target)
        print(f"subject{s}: unpacked", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["pack"] and len(sys.argv) == 3:
        pack(sys.argv[2])
        pack_labels(sys.argv[2])
    elif sys.argv[1:2] == ["unpack"] and len(sys.argv) in (2, 3):
        check_labels(*sys.argv[2:3])
        unpack(*sys.argv[2:3])
    else:
        sys.exit(__doc__)
