import json

from proctor_assist import env


def test_new_run_dir_writes_env_json(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "ROOT", tmp_path)
    run_dir = env.new_run_dir("t")
    info = json.loads((run_dir / "env.json").read_text())
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.startswith("t_")
    assert info["python"] and info["packages"]
