"""Download the OEP files that the project uses, and check them against the manifest (D11).

Only READ_ME.txt, the 24 gt.txt files, and the 24 webcam videos are downloaded. Audio and the
wearcam are out of scope (PLAN §2). Run: python -m proctor_assist.download
"""
import hashlib
import sys
import zipfile
from pathlib import Path

from proctor_assist.env import REPO, ROOT

DATASET = "raajanwankhade/oep-dataset"
OUT = ROOT / "data" / "raw" / "oep"
MANIFEST = REPO / "manifests" / "oep_raw.sha256"
EXPECTED = sorted(["READ_ME.txt"] + [f"subject{i}/{f}" for i in range(1, 25) for f in ("gt.txt", "webcam.avi")])


def local_name(kaggle_name):
    """Map a Kaggle file name to its local path, or None if the project does not use the file.

    READ_ME.txt names the webcam video <username>1.avi and the wearcam <username>2.avi. The webcam
    becomes subjectN/webcam.avi, so no username reaches the repo, the logs, or the app.
    """
    folder, _, name = kaggle_name.removeprefix("OEP database/").rpartition("/")
    if name == "READ_ME.txt" and not folder:
        return name
    if name == "gt.txt":
        return f"{folder}/gt.txt"
    if name.endswith("1.avi"):
        return f"{folder}/webcam.avi"
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, hashes):
    """Write {relative path: sha256} in `sha256sum` format, with LF line endings for Linux tools."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{h}  {rel}\n" for rel, h in sorted(hashes.items())), newline="\n")


def read_manifest(path):
    return {rel: h for h, rel in (line.split(maxsplit=1) for line in path.read_text().splitlines())}


def list_files(api):
    """Return [(name, size in bytes)] for every file in the Kaggle dataset."""
    files, token = [], None
    while True:
        page = api.dataset_list_files(DATASET, page_token=token, page_size=200)
        files += [(f.name, f.total_bytes) for f in page.files]
        token = page.next_page_token
        if not token:
            return files


def fetch(api, kaggle_name, size, dest):
    """Download one file to dest, unless dest already has the size from the Kaggle listing."""
    if dest.exists() and dest.stat().st_size == size:
        return
    tmp = dest.parent / ".partial"  # Kaggle resumes an interrupted download from this folder
    api.dataset_download_file(DATASET, kaggle_name, path=str(tmp), quiet=False)
    got = tmp / Path(kaggle_name).name
    zipped = tmp / f"{got.name}.zip"
    if zipped.exists():  # Kaggle sends large files as <name>.zip
        with zipfile.ZipFile(zipped) as z:
            z.extract(got.name, tmp)
        zipped.unlink()
    if not got.exists():
        sys.exit(f"Expected {got.name} from Kaggle, got {[p.name for p in tmp.iterdir()]}")
    got.replace(dest)
    tmp.rmdir()
    if dest.stat().st_size != size:
        sys.exit(f"{dest}: {dest.stat().st_size} bytes, but the Kaggle listing says {size}")


def main():
    from kaggle import api  # importing kaggle logs in, so tests can import this module without a token

    api.authenticate()  # the import hides a failed login. This call shows the reason.
    files = sorted((rel, n, s) for n, s in list_files(api) if (rel := local_name(n)))
    rels = [rel for rel, _, _ in files]
    if rels != EXPECTED:
        sys.exit(f"Unexpected Kaggle listing. Missing, extra, or duplicate: {sorted(set(EXPECTED) ^ set(rels))}")
    for i, (rel, name, size) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {rel}", flush=True)
        fetch(api, name, size, OUT / rel)

    hashes = {rel: sha256(OUT / rel) for rel in EXPECTED}
    if not MANIFEST.exists():
        write_manifest(MANIFEST, hashes)
        print(f"Wrote {MANIFEST}. Commit it, so that later runs check their files against it.")
        return
    expected = read_manifest(MANIFEST)
    bad = sorted(p for p in expected.keys() | hashes.keys() if expected.get(p) != hashes.get(p))
    if bad:
        sys.exit("These files differ from the manifest:\n" + "\n".join(bad))
    print(f"All {len(hashes)} files in {OUT} match {MANIFEST}.")


if __name__ == "__main__":
    main()
