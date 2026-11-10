# Project Plan: Cheating Detection for Online Exam Proctors

This plan fixes the **direction, goals, and constraints**. Implementation details (file
layout, exact hyperparameters, library choices, phase boundaries) are open for discussion
and are decided while building. Record decisions in `docs/decisions.md`.

Why this topic was chosen: `docs/decisions.md` D9.

---

## 1. Goal

Build and evaluate a deep learning system that watches the webcam video of online exam
takers and **flags moments for a proctor to review**. The system marks moments. The proctor
decides. The system never labels a person as a cheater.

The project must show whether temporal and landmark-based models detect more cheating
events than the per-frame method from prior work. The comparison holds the false-alarm rate
at a level that a proctor can accept. The project also measures how many exam takers one
CPU can monitor in real time.

**Target users:** exam proctors (lecturers, teachers) who supervise online exams through
webcams.
**Course:** COMP6826001 Deep Learning, BINUS (final project). The guideline is the
authority on what must be delivered.

## 2. Scope boundaries

In scope:
- Webcam video only.
- The visible cheating types in OEP: reading text, talking to a person in the room, a phone
  call, and phone use. The final class mapping is an owner decision at milestone 2.
- One experiment track (§5), with a reproduced paper method as the reference and the app
  fallback.
- A proctor dashboard app (§7).
- A practical evaluation without user testing (§8), subject to lecturer approval.

Out of scope (decided, do not reintroduce without discussion):
- Audio and the second (wearable) camera of OEP.
- Face recognition or identity verification.
- Screen or browser monitoring (the OEP type "use the Internet").
- Offline exams in a classroom (CCTV).
- LLM API features.
- Storage of video or frames by the app.

## 3. Research questions

1. Does a deep learning model detect cheating better than a simple head-turn rule without
   training? (R0 vs P1)
2. With a small dataset, how much does pretraining matter? (B1 vs P1)
3. Does temporal context over a window of a few seconds help, compared to one frame at a
   time? (P1 vs E1, E3)
4. Pixels or landmarks: what is the trade-off between detection quality and CPU cost?
   (E1 vs E2)
5. How well do models that learn mostly from actors generalize to real exam takers?
   (protocol A, results per subject type)
6. At what alarm threshold is the false-alarm rate acceptable, and how many exam takers can
   one laptop CPU monitor in real time?
7. How much does each model lose under low light, blur, compression, and a lower frame
   rate?

## 4. Data and ground truth

- **Dataset:** MSU Online Exam Proctoring (OEP) dataset, from Atoum et al., "Automated
  Online Exam Proctoring", IEEE Transactions on Multimedia 19(7), 2017. Cite this paper.
  Kaggle mirror: `raajanwankhade/oep-dataset`, 12.08 GB for all files (Kaggle file
  listing). The license on Kaggle is "Unknown". The original MSU page asks for a citation and
  names no license (D11).
- **Content, as the OEP paper reports it:** 24 subjects, all students at Michigan State
  University. 15 actors pretended to take the exam and cheated freely. 9 subjects took a
  real exam, and the proctor induced the cheating (talking, walking up, handing a book).
  Sessions are 17 min on average. There are 569 cheating instances of 5 types, and about
  20% of the video time is cheating.
- **Files per subject:** `gt.txt` (cheating intervals: start, end, type), one audio file,
  and two videos. `READ_ME.txt` says that `<username>1.avi` is the webcam above the
  monitor and `<username>2.avi` is the wearcam on the eyeglasses. The project downloads only
  the webcam videos (3.93 GB in total, Kaggle file listing), `gt.txt`, and `READ_ME.txt`
  (D11).
- **What the label files hold (milestone 2):** 565 rows with the type codes 1, 2, 3, 5, and
  6, not 1 to 5 as in the paper. D12 gives the meaning of the codes.
  `results/eda/summary.md` gives the statistics.
- **Ground truth:** the OEP intervals. The mapping from the 5 OEP types to the project
  classes is an owner decision at milestone 2. Proposal: the 4 classes of the reference
  paper (no cheat, read text, ask a person, call a friend). Intervals of the type "use the
  Internet" are masked: they are left out of training and evaluation, not labeled "no
  cheat".
- **Limitations to state in the report:** most cheating is acted. All data comes from one
  university setting in 2015, and 24 people is a small sample.

Non-negotiable data rules:
- Split by subject **before** any frame or window is extracted. No subject appears in two
  splits. A test checks this rule and fails on any subject in two splits.
- Anything fitted (normalization statistics, thresholds) comes from training or validation
  subjects only, never from test subjects.
- Alarm thresholds are chosen on validation subjects only.

## 5. Experiment design (single track)

| ID | Configuration | Variable | Role |
|---|---|---|---|
| R0 | Head-turn rule on face landmarks (MediaPipe), no training | (none) | Reference without training |
| B1 | Small CNN per frame, trained from scratch | (none) | Deep learning baseline |
| P1 | MobileNetV2 (ImageNet) per frame, with the configuration of the reference paper | Initial weights: random → ImageNet | Proven method and app fallback |
| E1 | MobileNetV2 frame features → GRU over a window of 2–4 s | One frame → a window | New: temporal context |
| E2 | Face, hand, and body landmarks (MediaPipe) → small GRU | Input: pixels → landmarks | New: CPU cost and scalability |
| E3 (optional) | Video model pretrained on Kinetics (for example X3D-S), fine-tuned | Pretraining: images → video | New, only if time and GPU allow |

**Reference paper for P1:** "Educational Innovation Faced with COVID-19: Deep Learning for
Online Exam Cheating Detection", Education Sciences 13(2), 194, 2023
(doi 10.3390/educsci13020194). An earlier version is in JITSI 2022
(doi 10.30630/jitsi.3.4.97). The paper reports MobileNetV2, 4 classes, at most 20 epochs,
early stopping with patience 10, learning rate 0.0001, batch size 16, and 5 dense layers.
It trains on 14 actors (9,000 frames), tests on the 9 real exam takers and 1 actor (7,000
frames), and reports F1 84.52%. These are the paper's numbers, not ours.

**Second reference, not directly comparable:** CHEESE (IEEE TCDS 2024, arXiv 2402.06107).
It reports a frame-level AUC of 87.58% on a re-organized OEP subset of 69 normal and 65
abnormal videos.

Key requirements:
- Every model gives one **suspicion score from 0 to 1 per window**: 1 − P(no cheat). The
  app reads only this score, so a model can change without a change to the app.
- The variables in the table are fixed. The architecture inside a scenario can change. Each
  change goes into `docs/decisions.md` with its reason (D10).
- Order: P1 first, so the app has a working model early. Then R0, B1, E1, and E2. E3 last.
- Trained models (B1, P1, E1, E2, E3) log the train and validation loss and macro F1 per
  epoch, so overfitting and convergence can be analyzed.
- Sanity check before full runs: the loss decreases, and the model beats a majority-class
  baseline.
- The class mapping, the window length, and the frame sampling rate are decided at
  milestones 2 and 3.

## 6. Evaluation

**Two split protocols, both by subject:**
- **Protocol A (comparison with the paper):** train on 14 actors, test on the 9 real exam
  takers and 1 actor. The paper does not name the actor in its test set, so our
  reproduction can differ from the paper.
- **Protocol B (main):** subject-grouped cross-validation, for example 4 folds of 6
  subjects, with actors and real exam takers in each fold. Report the mean and the standard
  deviation over the folds.

**Metrics:**
- Frame and window level: macro F1, F1 per class, confusion matrix, and ROC-AUC for cheat vs
  no cheat. These allow a comparison with the reference papers.
- **Event level (main metric for the app):** the share of cheating events detected vs the
  number of false alarms per hour. Also the time from the start of an event to the alarm. A
  tolerance of a few seconds applies at event boundaries. The value is decided at
  milestone 6.
- Compute: parameters, training time, CPU latency per window, memory, and the number of exam
  takers that one laptop CPU can monitor in real time.
- Changed conditions: low light, blur, compression, and a lower frame rate, applied to test
  video. Also a qualitative test on a laptop webcam with team members who consent. Nothing
  is stored.

**Analysis the guideline expects:**
- Training curves and behavior.
- Model comparison with compute cost (parameters, training time, latency, memory).
- Error analysis with patterns (per class, actors vs real exam takers).
- Results under changed conditions, scalability, and limitations.
- A justified choice of the deployment model.

## 7. Application

Proctor dashboard:
- A 3×3 grid plays the webcam videos of the 9 real exam takers from the test split, like a
  live feed. One tile can show the laptop webcam instead.
- Each tile shows the suspicion score over time. When the score passes the threshold, the
  tile is enlarged and the time span goes into a review list.
- The proctor replays the span and decides. The app shows no "cheater" label.
- Video is processed in memory and never stored.

Model choice for the app: keep the models that run in real time on CPU and have an
acceptable false-alarm rate on validation subjects. From these, take the one that detects
the most events. **If no new model beats P1, the app uses P1.**

Constraints: the app reuses the evaluation pipeline code instead of a copy, and runs on CPU.
Hosting must be free and need no credit card. The host is chosen at milestone 8. The demo
can run on a laptop.

## 8. Practical evaluation (instead of user testing)

Guideline §17 allows an alternative practical evaluation with lecturer approval. Request
approval for:
1. Evaluation against the OEP labels on subjects that the model did not see in training.
2. The tests under changed conditions (§6).
3. Error analysis with patterns.
4. Compute and scalability measurements.
5. A scenario demo of the dashboard on test subjects. The report describes it as a demo, not
   as user feedback.

## 9. Ethics and real-world suitability (report material)

- A false alarm can lead to an unfair accusation. Hence the human review, no automatic
  decision, and false alarms per hour as a main metric.
- The EU AI Act lists AI systems "for monitoring and detecting prohibited behaviour of
  students during tests" as high-risk (Annex III, point 3(d)).
- In Indonesia, UU 27/2022 on personal data protection treats biometric data as specific
  personal data (Pasal 4 ayat 2). The project does no face recognition and stores no video.
  Check the official text of the law before the report quotes it.
- The dataset is acted and comes from one lab in 2015. The results do not prove the
  performance in real exams.

## 10. Milestones (order, not a schedule)

1. Setup: the GitHub repo (D1), the OEP download script, and the local-first workflow
   (D8).
2. EDA: read `READ_ME.txt` and `gt.txt`, check the webcam file. The owner decides the
   class mapping.
3. Preprocessing: splits for protocols A and B, frame and window extraction, leakage test.
4. P1 reproduction (protocol A) and R0. From here, the app has a fallback model.
5. B1, E1, and E2 training (protocols A and B). E3 if time allows.
6. Event-level evaluation, thresholds on validation subjects, tests under changed
   conditions.
7. Error analysis, compute and scalability measurements, choice of the deployment model.
8. Dashboard app and deployment.
9. Practical evaluation (§8).
10. Figures, README, report, AI usage log finalized.

## 11. Risks

| Risk | Plan |
|---|---|
| Small dataset, large variance between splits | Protocol B, standard deviations, honest reporting |
| The reference paper does not fully describe its class mapping | Check the data first, decide the mapping, record it in `docs/decisions.md` |
| Label boundaries are not exact | Tolerance at event boundaries in the event metrics |
| License "Unknown" on Kaggle | Cite the original paper. Ask the lecturer |
| E3 is too heavy for Colab | E3 is optional. The core experiments do not need it |
| New models do not beat P1 | The app uses P1. The failed experiments are analysis material |

## 12. Guideline coverage

| Guideline requirement | Where it is covered |
|---|---|
| Real-world problem, stakeholder | §1 |
| Data description, preparation, bias, limits | §4 |
| DL model construction, pretrained explanation | B1 (from scratch), P1 (pretrained), E1–E3 |
| Baseline + ≥2 scenarios with clear variables | §5 |
| Metrics, training analysis, comparison, errors | §6 |
| Real-world suitability | §6 changed conditions, compute, scalability, §9 |
| Working application | §7 |
| Practical evaluation | §8 (needs lecturer approval) |
| AI usage log and integrity | `docs/ai_usage_log.md`, `CLAUDE.md` |
