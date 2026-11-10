import zipfile
from pathlib import Path

from proctor_assist import download as dl


def test_local_name_keeps_only_readme_gt_and_webcam():
    p = "OEP database/"
    # Made-up usernames. "user1" ends in a digit, like some real ones.
    assert dl.local_name(p + "READ_ME.txt") == "READ_ME.txt"
    assert dl.local_name(p + "subject3/gt.txt") == "subject3/gt.txt"
    assert dl.local_name(p + "subject3/user11.avi") == "subject3/webcam.avi"
    assert dl.local_name(p + "subject3/user12.avi") is None  # wearcam
    assert dl.local_name(p + "subject3/user1.wav") is None  # audio


def test_manifest_round_trip_and_changed_byte(tmp_path):
    f = tmp_path / "webcam.avi"
    f.write_bytes(b"abc")
    digest = dl.sha256(f)
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # SHA-256 test vector

    m = tmp_path / "m.sha256"
    dl.write_manifest(m, {"subject1/webcam.avi": digest})
    assert m.read_bytes() == f"{digest}  subject1/webcam.avi\n".encode()  # sha256sum format, LF only
    assert dl.read_manifest(m) == {"subject1/webcam.avi": digest}

    f.write_bytes(b"abd")  # same size, one byte changed
    assert dl.sha256(f) != digest


def test_fetch_unzips_and_renames(tmp_path):
    class FakeApi:  # behaves like Kaggle for a large file: it sends <name>.zip
        def dataset_download_file(self, dataset, file_name, path, quiet):
            Path(path).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(Path(path) / "user11.avi.zip", "w") as z:
                z.writestr("user11.avi", b"video")

    dest = tmp_path / "subject3" / "webcam.avi"
    dest.parent.mkdir()
    dl.fetch(FakeApi(), "OEP database/subject3/user11.avi", 5, dest)
    assert dest.read_bytes() == b"video"
    assert not (dest.parent / ".partial").exists()
