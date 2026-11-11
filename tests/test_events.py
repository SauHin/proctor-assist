import numpy as np
import pandas as pd

from proctor_assist import events

# All scores in this file are made up for the tests. They are not results.


def _scores(subject, suspicion, target, t=None):
    t = range(len(target)) if t is None else t
    return pd.DataFrame({"subject": subject, "t": t, "target": target, "suspicion": suspicion})


def _count(scores, threshold=0.5):
    return events.count(events.prepare(scores), threshold)


def test_runs_break_at_a_false_second_and_at_a_missing_second():
    starts, ends = events.runs([0, 1, 2, 4, 5, 6, 8], [1, 1, 0, 1, 1, 1, 1])
    assert starts.tolist() == [0, 4, 8] and ends.tolist() == [1, 6, 8]
    assert [len(x) for x in events.runs([0, 1], [0, 0])] == [0, 0]


def test_the_tolerance_counts_at_both_ends_of_an_event():
    target = [0] * 10 + [1] * 3 + [0] * 10  # event at seconds 10 to 12
    for alarm_second, hit in [(14, True), (15, False), (8, True), (7, False)]:
        suspicion = np.zeros(len(target))
        suspicion[alarm_second] = 1
        n, _ = _count(_scores(1, suspicion, target))
        assert (n["detected"], n["false_alarms"]) == ((1, 0) if hit else (0, 1))


def test_the_delay_is_from_the_event_start():
    target = [0] * 10 + [1] * 3 + [0] * 17 + [1] * 11 + [0] * 5  # events at 10 to 12 and 30 to 40
    suspicion = np.zeros(len(target))
    suspicion[[8, 33]] = 1
    n, delays = _count(_scores(1, suspicion, target))
    assert n["detected"] == 2 and n["false_alarms"] == 0 and delays == [-2, 3]


def test_an_alarm_over_the_whole_exam_gives_a_false_alarm_for_each_gap():
    target = [0] * 10 + [1] * 3 + [0] * 10 + [1] * 3 + [0] * 10  # 2 events, 3 parts outside
    n, _ = _count(_scores(1, np.ones(len(target)), target))
    assert n["detected"] == 2 and n["false_alarms"] == 3


def test_a_missing_second_splits_an_event_and_an_alarm():
    t = [0, 1, 2, 3, 5, 6, 7, 8]  # second 4 is masked
    n, _ = _count(_scores(1, [0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0], t))
    assert n == {"events": 1, "detected": 0, "false_alarms": 1, "seconds": 8}


def test_count_and_rates_over_subjects():
    a = _scores(1, [0, 0, .9, .9, 0, 0, 0, 0, 0, 0, 0, .9], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    b = _scores(2, [0] * 12, [0] * 5 + [2] * 3 + [0] * 4)
    n, delays = _count(pd.concat([a, b]))
    assert n == {"events": 2, "detected": 1, "false_alarms": 1, "seconds": 24} and delays == [0]
    assert events.rates(n) == {"detected_share": 0.5, "false_alarms_per_hour": 150.0}


def test_choose_takes_the_most_detections_within_the_budget():
    curve = pd.DataFrame(
        {"threshold": [0.1, 0.5, 0.7, 0.9, np.inf], "detected_share": [1, 0.8, 0.6, 0.6, 0], "false_alarms_per_hour": [9, 3, 2, 1, 0]}
    )
    assert events.choose(curve, budget=2) == 0.9  # 0.7 and 0.9 tie, the higher threshold wins
    assert events.choose(curve, budget=0) == np.inf


def test_the_threshold_does_not_depend_on_the_test_scores():
    rng = np.random.default_rng(0)
    target = (np.arange(3600) % 60 < 10).astype(int)
    val = _scores(1, rng.random(3600) * 0.5 + target * 0.4, target)
    test = _scores(2, rng.random(3600), target)
    row, _ = events.evaluate(val, test)
    row2, _ = events.evaluate(val, test.assign(suspicion=1 - test.suspicion))
    assert row["threshold"] == row2["threshold"] and row["val_false_alarms_per_hour"] <= events.BUDGET
