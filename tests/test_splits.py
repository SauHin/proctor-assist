import pytest

from proctor_assist import labels, splits

REAL = sorted(labels.REAL)
ACTORS = sorted(set(labels.SUBJECTS) - labels.REAL)


def test_check_fails_on_a_subject_in_two_splits():
    with pytest.raises(ValueError, match="7"):
        splits.check({"train": [1, 7], "val": [7], "test": [3]})


def test_committed_splits_follow_the_seed_and_have_no_subject_in_two_splits():
    s = splits.load_splits()
    rare = s["rare_code_subjects"]
    # The file must be the output of the code. A subject that was moved by hand fails here.
    assert s["protocol_a"] == splits.protocol_a(ACTORS, REAL, rare)
    assert s["protocol_b"] == splits.protocol_b(ACTORS, REAL, rare)
    for split in [s["protocol_a"], *s["protocol_b"]]:
        splits.check(split)
        assert sorted(split["train"] + split["val"] + split["test"]) == list(labels.SUBJECTS)


def test_protocol_a_trains_on_actors_and_tests_on_real_exam_takers_and_one_rare_code_actor():
    s = splits.load_splits()
    a, rare = s["protocol_a"], set(s["rare_code_subjects"])
    assert set(a["train"]) | set(a["val"]) <= set(ACTORS)
    (test_actor,) = set(a["test"]) - set(REAL)
    assert test_actor in rare and set(REAL) <= set(a["test"])
    assert len(set(a["val"]) & rare) == 1


def test_protocol_b_folds_have_both_groups_and_one_rare_code_actor():
    s = splits.load_splits()
    tests = [set(f["test"]) for f in s["protocol_b"]]
    assert sorted(x for t in tests for x in t) == list(labels.SUBJECTS)  # each subject is tested once
    for t in tests:
        assert len(t) == 6 and t & set(REAL) and t & set(ACTORS)
        assert len(t & set(s["rare_code_subjects"])) == 1


def test_rare_code_subjects_match_the_labels():
    if not labels.OEP.exists():
        pytest.skip("OEP data is not downloaded")
    df = labels.load_labels()
    assert sorted(df.subject[df.code == splits.RARE_CODE].unique()) == splits.load_splits()["rare_code_subjects"]
