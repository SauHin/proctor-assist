# PROPOSAL: class mapping and label cleaning

**Status: PROPOSAL. An AI wrote this draft on 2026-11-10. It is not a decision.**
The owner decides each point and writes the final version in `docs/decisions.md`. Until then,
no code uses this file.

Sources of every number: `results/eda/summary.md` and the CSV files next to it. The meaning of
the type codes is in `docs/decisions.md` (D12).

## What the data shows

| Code | Meaning (D12) | Intervals | Subjects | Seconds | In real exam takers |
|---|---|---|---|---|---|
| 1 | Text on paper | 192 | 22 | 1441 | 69 intervals, 7 subjects |
| 2 | Talk to a person in the room | 285 | 23 | 3645 | 193 intervals, 9 subjects |
| 3 | Use the Internet | 52 | 9 | 1533 | none |
| 5 | Ask a friend over the phone (inferred) | 15 | 4 | 110 | none |
| 6 | Use a phone (inferred) | 20 | 8 | 452 | 1 interval of 4 s |

- The 9 real exam takers have almost only the codes 1 and 2. All other codes come from actors.
- What a webcam frame shows, from 8 frames per code in `data/eda/frames/`. This is an AI
  observation, and the owner must check it. Code 1 looks down or to the side. Code 2 talks,
  and a second person is sometimes at the edge of the frame. Code 3 looks at the screen, as in
  normal exam work. Code 5 holds a phone at the ear. Code 6 looks down, and the phone is not
  in the frame.

## Decision 1: project classes

The model gives a suspicion score of 1 - P(no cheat) (D10). Thus the class mapping changes the
per-class F1 and the training signal. It does not change which seconds count as cheating,
except for a code that is masked.

| Option | Classes | For | Against |
|---|---|---|---|
| **A (recommended)** | 0 no cheat. 1 looks at an object: codes 1 and 6. 2 talks to a person: code 2. 3 phone call: code 5. Code 3 masked. | 4 classes, as in the reference paper. Each class is something that the webcam can see. Phone use stays a cheating type (PLAN §2). | Class 1 is wider than the paper's "read text". The paper does not say where it put phone use. |
| B | As A, but code 6 is a class of its own (5 classes). | No assumption about how code 6 looks. | 20 intervals in 8 subjects, and 1 interval in the real exam takers. The F1 of this class is not reliable, and it pulls macro F1. Not the paper's 4 classes. |
| C | As A, but code 6 joins code 5 as one "phone" class. | A clear name. More data for class 3. | One class holds two different poses: phone at the ear, and looking down. The second pose looks like class 1. |
| D | As A, but code 6 is masked. | The classes equal the paper's list. | Phone use is in scope (PLAN §2). The system then never learns it and is never tested on it. |

Reason for A: the project uses the webcam only. If the webcam cannot tell a class apart, the
class is not useful. Before you accept A, compare `data/eda/frames/code1.jpg` with `code6.jpg`
yourself.

Consequence for protocol A (PLAN §6): the test set is the 9 real exam takers and 1 actor. The
real exam takers have no code 5. Thus the "phone call" class in the test set comes from that
one actor only. The choice of this actor changes the F1 of that class.

## Decision 2: the Internet type (code 3)

PLAN §4 already proposes a mask: these seconds are left out of training and evaluation, and
they are not labeled "no cheat". The data supports this. Code 3 has 52 intervals in 9 actors,
and the frames look like normal exam work.

Open detail: code 3 shares time with code 1 (5 pairs, 50 s) and with code 2 (3 pairs, 14 s).
Recommendation: where a visible code and code 3 share a second, the visible code wins. Only
the seconds that code 3 covers alone are masked.

## Decision 3: rows with a problem

The raw files stay as they are, because they must match the manifest (D11). A correction goes
into a small file that the owner writes, for example `configs/label_fixes.csv`. Its columns
are subject, line, action, start, end, code, and reason. The code of milestone 3 applies this
file, and a test checks it.

| Row | Problem | Options | Recommendation |
|---|---|---|---|
| subject7 line 14: `0900 00903 6` | The end time has 5 digits. | (a) Correct it to `0900 0903 6`. (b) Drop the row. (c) Mask these seconds. | (a). The start is 09:00, and one extra zero is the simplest explanation. (b) labels 3 s of cheating as "no cheat". |
| subject7 lines 18, 19, 20 | They repeat lines 11, 3, and 12 exactly. | (a) Keep one copy. | (a). No information is lost. |
| subject8 line 3: `0112 1019 3` | 547 s long. It shares time with 11 other rows of code 3. The next row starts at 01:25. `0112 0119` is possible. | (a) Keep it. (b) Correct it after you watch the video. (c) Drop it. | If code 3 is masked, (a). Then a wrong long row only removes data. It never adds a wrong label. If you want that data back, watch 01:12 to 10:19. |
| subject1 line 6: `0230 0336 2` | It shares time with 2 other rows of code 2. The next row starts at 02:37. `0230 0236` is possible. | (a) Keep it. (b) Correct it after you watch the video. (c) Mask 02:36 to 03:36. | Watch subject1 from 02:30 to 03:36 (about one minute), then choose (a) or (b). If you cannot decide, choose (c). Code 2 is a visible class, so a wrong row here adds wrong labels. |

Also: 27 rows start before the row above them. This needs no decision. The code sorts the
rows.

## Decision 4: intervals of different codes that share time

74 pairs of intervals with different codes share time. 65 of them are code 1 with code 2, in 8
subjects, with 264 s in total. These subjects read text while they talk. The timeline figure
(`results/eda/timeline.png`) shows it for subject18: short code 1 intervals inside long code 2
intervals. One more pair is code 5 with code 6 (4 s).

For the event metric, this does not matter: a second with any visible code is a cheating
second. A classifier with one label per frame needs a rule.

| Option | Rule | For | Against |
|---|---|---|---|
| **A (recommended)** | The class with fewer intervals wins: phone call, then looks at an object, then talks to a person. | Simple. No data is lost. The rare classes keep all their frames. The short, specific action wins over the long one around it. | One of two true labels is dropped. |
| B | Shared seconds are masked for the class loss and the class metrics. They still count as cheating for the event metric. | No arbitrary choice. | About 264 s of cheating frames leave the training set. Two kinds of mask make the code harder to explain. |
| C | Multi-label output: one probability per class. | Matches reality. | P1 no longer follows the reference paper. The suspicion score needs a new definition (D10). |

## Decision 5: is the end second part of the interval?

If the times are instants, `0221 0223` is 2 s. If the row means the seconds 141, 142, and
143, it is 3 s. The difference is 564 s over all rows (7181 s against 7745 s). The paper reports
7235 s, which is between the two. No interval has the same start and end, which fits the
"instants" reading. Some rows start one second after the row above ends, which fits the other
reading.

Recommendation: read the times as instants, so an interval is [start, end). The effect is at
most 1 s per interval. The tolerance at event boundaries (PLAN §6, milestone 6) is larger.

## Before you decide: checks that only the owner can do

1. Open the five contact sheets in `data/eda/frames/`. Check the statements about code 5 and
   code 6 in D12, and compare code 1 with code 6.
2. Watch subject1 from 02:30 to 03:36, and (optional) subject8 from 01:12 to 10:19.
3. Read section IV of the final IEEE version of the OEP paper (IEEE Xplore through BINUS).
   Check that the 5 types are the same as in the authors' version (D12).
4. Optional: ask the dataset author about the codes 5 and 6 (address in `READ_ME.txt`).

## Template for the owner's entry in `docs/decisions.md`

Copy this block, fill it in, and remove the options that you do not choose.

```
## D14. Class mapping and label cleaning (owner, <date>)

- Classes: <A / B / C / D, or your own list: class id, name, OEP codes>
- Code 3: <masked / other>. Where code 3 shares time with a visible code: <rule>
- subject7 line 14: <correct to ... / drop / mask>
- subject7 duplicate rows: <keep one copy / other>
- subject8 line 3: <keep / correct to ... / drop>. Checked by: <what you watched>
- subject1 line 6: <keep / correct to ... / mask>. Checked by: <what you watched>
- Different codes that share time: <A / B / C>
- End second: <[start, end) / end second included>
- Reason for each choice: <...>
- What I checked myself: <contact sheets, video moments, IEEE version>
```
