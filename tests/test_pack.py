import pytest

from proctor_assist import frames, pack


def test_pack_then_unpack_gives_the_same_bytes_and_a_changed_zip_fails(tmp_path, monkeypatch):
    src, drive, manifest = tmp_path / "laptop", tmp_path / "drive", tmp_path / "m.sha256"
    for s in (1, 2):
        folder = src / f"subject{s}"
        folder.mkdir(parents=True)
        (folder / "000000.jpg").write_bytes(bytes([s]) * 10)
        (folder / "_done.json").write_text("{}")
    monkeypatch.setattr(frames, "OUT", src)
    hashes = pack.pack(drive, subjects=(1, 2), manifest=manifest)
    assert pack.pack(drive, subjects=(1, 2), manifest=manifest) == hashes  # same bytes on a second run

    monkeypatch.setattr(frames, "OUT", tmp_path / "colab")
    pack.unpack(drive, subjects=(1, 2), manifest=manifest)
    assert (tmp_path / "colab" / "subject2" / "000000.jpg").read_bytes() == bytes([2]) * 10

    monkeypatch.setattr(frames, "OUT", tmp_path / "colab2")
    zip1 = drive / pack.ZIP_DIR / "subject1.zip"
    zip1.write_bytes(zip1.read_bytes()[:-1] + b"x")  # one changed byte, as after a broken upload
    with pytest.raises(SystemExit, match="manifest"):
        pack.unpack(drive, subjects=(1, 2), manifest=manifest)
