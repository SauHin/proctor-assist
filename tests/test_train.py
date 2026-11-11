import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from proctor_assist import data, metrics, models, train


def test_p1_head_has_the_sizes_of_the_paper():
    model = models.P1(pretrained=False)
    x = torch.zeros(2, 3, 224, 224)
    assert model.head[:4](model.features(x)).shape == (2, 400)  # convolution, ReLU, pooling, flatten
    assert sum(isinstance(layer, nn.Linear) for layer in model.head) == 6  # 5 dense layers and the output
    assert model(x).shape == (2, 4)


def test_expand_gives_all_frames_for_training_and_the_last_frame_for_a_score():
    windows = pd.DataFrame({"subject": [1, 1], "t": [0, 3], "target": [0, 2]})
    train_rows = data.expand(windows, train=True)
    assert train_rows.frame.tolist() == [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]
    assert train_rows.target.tolist() == [0] * 5 + [2] * 5
    assert data.expand(windows, train=False).frame.tolist() == [4, 19]


def test_report_and_suspicion_on_known_values():
    probs = np.array([[0.9, 0.1, 0, 0], [0.2, 0.7, 0.1, 0], [0.6, 0.1, 0.3, 0], [0.1, 0, 0, 0.9]])
    r = metrics.report(np.array([0, 1, 2, 3]), probs)
    assert metrics.suspicion(probs).tolist() == pytest.approx([0.1, 0.8, 0.4, 0.9])
    assert r["accuracy"] == 0.75  # the row of class 2 is predicted as class 0
    assert r["confusion_matrix_rows_true_cols_pred"][2] == [1, 0, 0, 0]
    assert r["macro_f1"] == pytest.approx((2 / 3 + 1 + 0 + 1) / 4)
    assert r["roc_auc_cheat"] == 1.0  # each cheating row has a higher suspicion than the no-cheat row
    assert r["baseline_accuracy"] == 0.25


class Tiny(nn.Module):
    def __init__(self, dense_layers, units, pretrained):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    def forward(self, x):
        return self.net(x)


def test_run_writes_the_run_files_resumes_and_stops_early(tmp_path, monkeypatch):
    # A tiny model and made-up 8 x 8 frames, so that the loop runs in a moment without data.
    table = pd.DataFrame({"subject": 1, "t": range(40), "target": [0, 1, 2, 3] * 10, "frame": range(40)})
    monkeypatch.setitem(models.MODELS, "tiny", Tiny)
    monkeypatch.setattr(train, "frame_table", lambda subjects, train: table)
    monkeypatch.setattr(data, "read_frame", lambda s, k: torch.full((3, 8, 8), k % 4 * 60, dtype=torch.uint8))
    cfg = {"model": "tiny", "protocol": "a", "seed": 0, "epochs": 3, "patience": 10, "batch_size": 8, "lr": 0.01,
           "dense_layers": 1, "units": 4, "pretrained": False, "balanced_sampling": True, "augment": None, "workers": 0}

    first = tmp_path / "first"
    first.mkdir()
    m = train.run(cfg, first)
    assert m["epochs_run"] == 3 and m["test"]["rows"] == 40
    for name in ("history.csv", "last.pt", "best.pt", "metrics.json", "scores_val.csv", "scores_test.csv"):
        assert (first / name).exists()
    history = pd.read_csv(first / "history.csv")
    assert history.epoch.tolist() == [1, 2, 3] and history.train_loss.iloc[-1] < history.train_loss.iloc[0]

    train.run(cfg | {"epochs": 5}, first)  # the same folder: the run continues at epoch 4
    assert pd.read_csv(first / "history.csv").epoch.tolist() == [1, 2, 3, 4, 5]

    second = tmp_path / "second"
    second.mkdir()
    stopped = train.run(cfg | {"epochs": 10, "patience": 1, "lr": 0.0}, second)  # no learning, so no better epoch
    assert stopped["epochs_run"] == 2 and stopped["best_epoch"] == 1


def test_per_subject_and_per_code_tables_on_toy_scores():
    from proctor_assist import evaluate

    scores = pd.DataFrame(
        {"part": "val", "subject": 1, "t": [0, 1, 2, 3], "target": [0, 0, 1, 1], "p0": [0.9, 0.8, 0.3, 0.6]}
    )
    row = evaluate.per_subject(scores).iloc[0]
    assert (row.seconds, row.cheat_seconds, row.roc_auc_cheat) == (4, 2, 1.0)
    assert row.mean_suspicion_cheat == pytest.approx(0.55) and row.group == "actor"

    labels = pd.DataFrame({"subject": [1, 1], "start_s": [2, 3], "end_s": [4, 4], "code": [1, 6]})
    codes = evaluate.per_code(scores, labels).set_index("code")
    assert codes.seconds.to_dict() == {0: 2, 1: 2, 6: 1}  # second 3 has the codes 1 and 6
    assert codes.share_suspicion_over_half[1] == 0.5 and codes.mean_suspicion[6] == pytest.approx(0.4)


def test_the_protocol_b_config_equals_the_protocol_a_config_and_picks_its_fold():
    import yaml

    from proctor_assist import splits
    from proctor_assist.env import REPO

    a = yaml.safe_load((REPO / "configs" / "p1_a.yaml").read_text())
    b = yaml.safe_load((REPO / "configs" / "p1_b.yaml").read_text())
    assert {k for k in a if a[k] != b[k]} == {"name", "protocol"}
    assert train.split_of(b | {"fold": 2}) == splits.load_splits()["protocol_b"][1]
    assert train.split_of(a) == splits.load_splits()["protocol_a"]


def test_summary_gives_one_row_per_run_and_the_mean_and_standard_deviation(tmp_path):
    import json

    from proctor_assist import evaluate

    part = {"macro_f1": 0.0, "baseline_macro_f1": 0.2, "accuracy": 0.5, "baseline_accuracy": 0.7, "roc_auc_cheat": 0.6}
    for k, f1 in ((1, 0.3), (2, 0.5)):
        folder = tmp_path / f"m_f{k}_x"
        folder.mkdir()
        metrics = {"epochs_run": 4, "best_epoch": 2, "train_seconds": 120, "val": part, "test": part | {"macro_f1": f1}}
        (folder / "metrics.json").write_text(json.dumps(metrics))
        rows = pd.DataFrame({"part": ["val", "test"], "subject": [k, k + 10], "roc_auc_cheat": [0.5, 0.9]})
        rows.to_csv(folder / "per_subject.csv", index=False)
    (tmp_path / "r0").mkdir()
    pd.DataFrame({"subject": [11, 12], "roc_auc_cheat": [0.7, 0.8]}).to_csv(tmp_path / "r0" / "per_subject.csv", index=False)

    evaluate.summary("m_f", results=tmp_path)
    table = pd.read_csv(tmp_path / "m_f_summary.csv").set_index("run")
    assert table.index.tolist() == ["m_f1_x", "m_f2_x", "mean", "std"]
    assert table.test_macro_f1["mean"] == pytest.approx(0.4) and table.test_macro_f1["std"] == pytest.approx(0.1414, abs=1e-4)
    assert table.train_minutes["mean"] == 2
    per = pd.read_csv(tmp_path / "m_f_test_per_subject.csv")
    assert per.subject.tolist() == [11, 12] and per.r0_roc_auc_cheat.tolist() == [0.7, 0.8]


def test_b1_is_small_and_build_gives_each_model_only_its_own_config_values():
    cfg = {"model": "b1", "name": "x", "lr": 0.1, "pretrained": True}  # B1 has no pretrained weights
    model = models.build(cfg)
    assert isinstance(model, models.B1) and model(torch.zeros(2, 3, 224, 224)).shape == (2, 4)
    assert sum(p.numel() for p in model.parameters()) < 500_000
    p1 = models.build({"model": "p1", "dense_layers": 2, "units": 8, "pretrained": True}, pretrained=False)
    assert sum(isinstance(layer, nn.Linear) for layer in p1.head) == 3


def test_the_other_per_frame_configs_differ_from_p1_only_in_their_variable():
    import yaml

    from proctor_assist.env import REPO

    load = lambda name: yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
    for protocol in ("a", "b"):
        p1, p1s, b1 = load(f"p1_{protocol}"), load(f"p1s_{protocol}"), load(f"b1_{protocol}")
        assert {k for k in p1 if p1[k] != p1s[k]} == {"name", "pretrained"} and p1s["pretrained"] is False
        assert {k for k in p1 if p1[k] != b1.get(k)} == {"name", "model", "dense_layers", "units", "pretrained"}
        assert b1["model"] == "b1" and set(b1) <= set(p1)


def test_e1_reads_the_15_frames_of_a_window_and_trains_on_stored_features(tmp_path, monkeypatch):
    from proctor_assist import features

    feats = {1: torch.arange(200.0).repeat(128, 1).T}  # frame k has the value k in each of its 128 features
    table = pd.DataFrame({"subject": 1, "t": range(2, 40), "target": [0, 1] * 19, "frame": 0})
    x, target = features.SequenceDataset(table, feats)[1]  # the window that ends with second 3
    assert x.shape == (15, 128) and x[:, 0].tolist() == list(range(5, 20)) and target == 1
    assert models.build({"model": "e1", "hidden": 8})(torch.zeros(4, 15, 128)).shape == (4, 4)

    monkeypatch.setattr(train, "frame_table", lambda subjects, train: table)
    monkeypatch.setattr(features, "load", lambda cfg, device: feats)
    cfg = {"model": "e1", "features": "p1", "protocol": "a", "seed": 0, "epochs": 2, "patience": 10, "batch_size": 8,
           "lr": 0.01, "hidden": 8, "balanced_sampling": True}
    assert train.run(cfg, tmp_path)["epochs_run"] == 2 and (tmp_path / "scores_test.csv").exists()


def test_the_e1_configs_keep_the_training_values_of_p1():
    import yaml

    from proctor_assist.env import REPO

    load = lambda name: yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
    for protocol in ("a", "b"):
        p1, e1 = load(f"p1_{protocol}"), load(f"e1_{protocol}")
        assert {k for k in e1 if e1[k] != p1.get(k)} == {"name", "model", "features", "hidden"}
        assert e1["features"] == "p1" and "augment" not in e1
