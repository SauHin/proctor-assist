import numpy as np
import pandas as pd

from proctor_assist.analysis import analyse, event_class


def test_event_class_majority_and_tie():
    assert event_class(np.array([2, 2, 1])) == 2
    assert event_class(np.array([2, 1])) == 1  # tie: looks at an object comes before talk in PRIORITY
    assert event_class(np.array([3, 1, 1, 2, 2])) == 1


def test_analyse_counts_events_and_false_alarms():
    t = np.arange(20)
    target = np.zeros(20, int)
    target[5:8] = 2  # one event, seconds 5 to 7
    sus = np.zeros(20)
    sus[[9, 15, 16]] = 0.9  # 9 is inside the tolerance, 15-16 is one false alarm
    scores = pd.DataFrame({"subject": 1, "t": t, "target": target, "suspicion": sus})
    ev, subj, fa = analyse(scores, 0.5, {(1, 6): {2}})
    assert ev.detected.tolist() == [True] and ev.delay_s.tolist() == [4] and ev.codes.tolist() == ["2"]
    assert ev["class"].tolist() == ["talks to a person"] and ev.length_s.tolist() == [3]
    assert fa[["start_s", "end_s", "peak_t"]].values.tolist() == [[15, 16, 15]]
    assert subj[["events", "detected", "false_alarms", "seconds"]].values.tolist() == [[1, 1, 1, 20]]
