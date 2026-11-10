"""Subject splits for protocols A and B (PLAN §6, D15). Run: python -m proctor_assist.splits

The run writes splits/subjects.json. Every later step reads this file, so the local machine and
Colab use the same subjects. A subject gets its split before any frame is extracted. The run
also writes results/splits/composition.csv: the labeled intervals per type code in each part.
"""
import json
import random

import pandas as pd

from proctor_assist.env import REPO
from proctor_assist.labels import REAL, SUBJECTS, load_labels

SEED = 0
FILE = REPO / "splits" / "subjects.json"
RARE_CODE = 5  # phone call. Only a few actors have it, so they are spread over the splits (D15).


def check(split):
    """Fail if a subject is in more than one of train, val, and test (data leakage rule)."""
    seen = [s for part in ("train", "val", "test") for s in split[part]]
    twice = sorted({s for s in seen if seen.count(s) > 1})
    if twice:
        raise ValueError(f"Subjects in more than one split: {twice}")
    return split


def protocol_a(actors, real, rare, n_val=3, seed=SEED):
    """Train and validate on actors. Test on the real exam takers and one actor.

    The test actor and one validation actor come from `rare`, the actors with the rare code.
    """
    rng = random.Random(seed)
    test_actor, val_rare = rng.sample(sorted(rare), 2)
    val = [val_rare, *rng.sample(sorted(set(actors) - set(rare)), n_val - 1)]
    train = set(actors) - {test_actor} - set(val)
    return check({"train": sorted(train), "val": sorted(val), "test": sorted([*real, test_actor])})


def protocol_b(actors, real, rare, n_folds=4, seed=SEED):
    """Cross-validation folds. Fold k is the test set, and the next fold is the validation set.

    The subjects are dealt like cards: the rare-code actors, then the other actors, then the real
    exam takers. Thus each fold gets its share of every group.
    """
    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]
    i = 0
    for group in (set(rare), set(actors) - set(rare), set(real)):
        members = sorted(group)
        rng.shuffle(members)
        for s in members:
            folds[i % n_folds].append(s)
            i += 1
    out = []
    for k in range(n_folds):
        v = (k + 1) % n_folds
        train = [s for j, fold in enumerate(folds) if j not in (k, v) for s in fold]
        out.append(check({"train": sorted(train), "val": sorted(folds[v]), "test": sorted(folds[k])}))
    return out


def load_splits():
    return json.loads(FILE.read_text())


def composition(splits, labels):
    """Count the subjects and the labeled intervals per type code in each part of each split."""
    valid = labels[labels.problem == ""]
    named = [("A", splits["protocol_a"]), *((f"B fold {k}", f) for k, f in enumerate(splits["protocol_b"], 1))]
    rows = []
    for name, split in named:
        for part in ("train", "val", "test"):
            counts = valid[valid.subject.isin(split[part])].code.value_counts()
            row = {"split": name, "part": part, "subjects": len(split[part]), "real": len(REAL & set(split[part]))}
            rows.append(row | {f"code_{c}": int(counts.get(c, 0)) for c in sorted(valid.code.unique())})
    return pd.DataFrame(rows)


def main():
    labels = load_labels()
    actors, real = sorted(set(SUBJECTS) - REAL), sorted(REAL)
    rare = sorted(int(s) for s in labels.subject[labels.code == RARE_CODE].unique())
    splits = {
        "seed": SEED,
        "rare_code": RARE_CODE,
        "rare_code_subjects": rare,
        "protocol_a": protocol_a(actors, real, rare),
        "protocol_b": protocol_b(actors, real, rare),
    }
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(splits, indent=2) + "\n", newline="\n")
    table = REPO / "results" / "splits" / "composition.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    composition(splits, labels).to_csv(table, index=False, lineterminator="\n")
    print(f"Wrote {FILE} and {table}")


if __name__ == "__main__":
    main()
