"""CPU load test of milestone 7 (D42): how many exam takers one laptop scores in real time.

Run on the laptop (MediaPipe, D25):
  python -m proctor_assist.capacity --runs "G:/My Drive/proctor-assist/runs"

A trial runs N video streams at the same time through the steps of the app: decode each video
frame (25 fps), then the steps of one model. The streams are spread over WORKERS processes, and
each process scores its streams one second at a time, in turn. A trial is real time if every
process scores SECONDS seconds of video of all its streams in at most SECONDS seconds. The search
doubles N until a trial fails, then halves the gap. The models are the protocol A copies, which
the app uses (D42). Writes results/capacity/trials.csv, summary.csv, and run.json.
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.io import decode_jpeg

from proctor_assist import frames, landmarks, r0
from proctor_assist.conditions import checkpoint
from proctor_assist.data import transform
from proctor_assist.env import REPO, ROOT, env_info
from proctor_assist.labels import OEP
from proctor_assist.models import build
from proctor_assist.splits import load_splits
from proctor_assist.windows import WINDOW_S

MODELS = ("r0", "b1", "p1", "e1", "e2")
SECONDS = 60  # seconds of video per stream in a trial
WORKERS = os.cpu_count()  # one process per logical core, 1 torch thread each
MAX_STREAMS = 256  # memory: about 14 GB of RAM on the laptop
LAST = frames.VIDEO_FPS - frames.STEP  # video frame in a second of the last stored frame (20)
OUT = REPO / "results" / "capacity"


def run_folder(model):
    (folder,) = sorted((REPO / "results").glob(f"{model}_a_2*"))  # the protocol A copy (D42)
    return folder


def load(model, places):
    """The model of a protocol A run, and for E1 also its P1 backbone, on the CPU."""
    def one(folder):
        net = build(yaml.safe_load((folder / "config.yaml").read_text()), pretrained=False)
        net.load_state_dict(torch.load(checkpoint(folder.name, places), map_location="cpu"))
        return net.eval()

    if model == "r0":
        return None, None
    folder = run_folder(model)
    backbone = one(run_folder("p1")) if model == "e1" else None
    return one(folder), backbone


def image(frame, tf):
    """A video frame as the model input of P1, B1, and E1: the resize and the JPEG step of D18."""
    ok, data = cv2.imencode(".jpg", frames.preprocess(frame), [cv2.IMWRITE_JPEG_QUALITY, frames.QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return tf(decode_jpeg(torch.from_numpy(data.reshape(-1))))[None]


class Scorer:
    """The steps of one model for one stream. `frame` returns P(no cheat) at the end of each second."""

    def __init__(self, model, net, backbone, tools, stages):
        self.model, self.net, self.backbone, self.tools, self.stages = model, net, backbone, tools, stages
        self.buffer = deque(maxlen=frames.FPS * WINDOW_S)
        self.poses = []
        self.tf = transform()

    def timed(self, stage, fn, *args):
        a = time.perf_counter()
        out = fn(*args)
        self.stages[stage] += time.perf_counter() - a
        return out

    @torch.no_grad()
    def frame(self, i, frame):
        stored, last = i % frames.STEP == 0, i % frames.VIDEO_FPS == LAST
        m = self.model
        if m == "r0" and last:
            self.poses.append(self.timed("mediapipe", r0.pose_row, self.tools, frame, i // frames.VIDEO_FPS))
            return 1 - self.timed("window", lambda: r0.suspicion(pd.DataFrame(self.poses))[-1])
        if m in ("p1", "b1") and last:
            x = self.timed("prepare", image, frame, self.tf)
            return self.timed("model", lambda: self.net(x).softmax(1)[0, 0].item())
        if m == "e1" and stored:
            x = self.timed("prepare", image, frame, self.tf)
            self.buffer.append(self.timed("model", lambda: self.backbone.head[:-1](self.backbone.features(x))[0]))
        if m == "e2" and stored:
            raw = landmarks.empty(1)
            self.timed("mediapipe", landmarks.detect, self.tools, frame, raw, 0)
            self.buffer.append(torch.from_numpy(self.timed("prepare", landmarks.frame_features, raw)[0]))
        if m in ("e1", "e2") and last and len(self.buffer) == self.buffer.maxlen:
            return self.timed("window", lambda: self.net(torch.stack(list(self.buffer))[None]).softmax(1)[0, 0].item())
        return None


def peak_memory_mb():
    """Peak working set of this process (Windows). shortcut: Windows only, as the laptop of D8."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    c = Counters(cb=ctypes.sizeof(Counters))
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return c.PeakWorkingSetSize / 2**20


def on_ac_power():
    """True if the laptop runs on the charger (Windows). The speed of a laptop CPU depends on it."""
    class Status(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ubyte) for n in ("ACLineStatus", "BatteryFlag", "BatteryLifePercent", "SystemStatusFlag")] + [
            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = Status()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ACLineStatus == 1


def _worker(model, subjects, places, seconds, barrier, out):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net, backbone = load(model, places)
    tools = {"r0": r0.landmarker, "e2": landmarks._tools}.get(model, lambda: None)()
    stages = Counter()
    streams = [(cv2.VideoCapture(str(OEP / f"subject{s}" / "webcam.avi")), Scorer(model, net, backbone, tools, stages))
               for s in subjects]
    latencies, scores = [], 0
    barrier.wait()
    start = time.perf_counter()
    for second in range(seconds):
        for cap, scorer in streams:
            for j in range(frames.VIDEO_FPS):
                a = time.perf_counter()
                ok, frame = cap.read()
                b = time.perf_counter()
                stages["decode"] += b - a
                if not ok:
                    raise SystemExit("A video ended before the trial.")
                if scorer.frame(second * frames.VIDEO_FPS + j, frame) is not None:
                    latencies.append(time.perf_counter() - b)  # from the decoded last frame to the score
                    scores += 1
    wall = time.perf_counter() - start
    out.put({"wall_s": wall, "streams": len(subjects), "stages": dict(stages), "latencies": latencies,
             "scores": scores, "peak_mb": peak_memory_mb()})


def trial(model, n, places, subjects, seconds=SECONDS):
    """One trial with n streams. Returns its row."""
    ctx = mp.get_context("spawn")
    workers = min(n, WORKERS)
    streams = [[subjects[s % len(subjects)] for s in range(n)][w::workers] for w in range(workers)]
    barrier, out = ctx.Barrier(workers), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(model, part, places, seconds, barrier, out)) for part in streams]
    for p in procs:
        p.start()
    results = [out.get(timeout=20 * seconds + 600) for _ in procs]
    for p in procs:
        p.join()
        if p.exitcode:
            raise SystemExit(f"A worker of {model} with {n} streams failed.")
    lat = np.concatenate([r["latencies"] for r in results]) * 1000
    stages = sum((Counter(r["stages"]) for r in results), Counter())
    video_s = n * seconds
    return {
        "model": model, "streams": n, "workers": workers, "seconds": seconds,
        "max_wall_s": max(r["wall_s"] for r in results), "real_time": max(r["wall_s"] for r in results) <= seconds,
        "scores": sum(r["scores"] for r in results),
        "latency_p50_ms": np.percentile(lat, 50) if len(lat) else np.nan,
        "latency_p95_ms": np.percentile(lat, 95) if len(lat) else np.nan,
        "peak_mb_sum": sum(r["peak_mb"] for r in results), "peak_mb_max_worker": max(r["peak_mb"] for r in results),
    } | {f"{k}_ms_per_video_s": 1000 * v / video_s for k, v in stages.items()}


def search(model, places, subjects):
    """Rows of the trials, from 1 stream up, until the largest real-time number is known."""
    rows = []
    def run(n):
        rows.append(trial(model, n, places, subjects))
        r = rows[-1]
        print(f"{model} {n} streams: {r['max_wall_s']:.1f} s for {SECONDS} s, real time {r['real_time']}", flush=True)
        return r["real_time"]

    lo, hi = 0, 1
    while hi <= MAX_STREAMS and run(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1 and 0 < lo < MAX_STREAMS:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if run(mid) else (lo, mid)
    return rows, lo


def compute_cost(model):
    """Parameters at scoring time and the training time of the protocol A copy."""
    if model == "r0":
        return {"parameters": 0, "train_seconds": 0.0, "train_device": "none"}
    folder = run_folder(model)
    m = json.loads((folder / "metrics.json").read_text())
    params = m["parameters"]
    if model == "e1":  # E1 also runs the P1 backbone
        params += json.loads((run_folder("p1") / "metrics.json").read_text())["parameters"]
    env = json.loads((folder / "env.json").read_text())
    return {"parameters": params, "train_seconds": m["train_seconds"], "train_device": env.get("gpu") or m["device"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", help="folder with the run folders of P1, B1, and E1, for example the Drive folder")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    info = env_info()  # before the outputs are written
    places = [Path(args.runs), ROOT / "runs"] if args.runs else [ROOT / "runs"]
    subjects = load_splits()["protocol_a"]["test"]  # the subjects of the app (D42)
    OUT.mkdir(parents=True, exist_ok=True)
    ac = on_ac_power()
    rows, summary = [], []
    for model in args.models:
        trials, best = search(model, places, subjects)
        rows += trials
        t = pd.DataFrame(trials).set_index("streams")
        one = t.loc[1]
        summary.append({"model": model, "max_real_time_streams": best} | compute_cost(model) | {
            "latency_p50_ms_1_stream": one.latency_p50_ms, "latency_p95_ms_1_stream": one.latency_p95_ms,
            "latency_p95_ms_at_max": t.loc[best].latency_p95_ms if best else np.nan,
            "peak_mb_max_worker_at_max": t.loc[best].peak_mb_max_worker if best else np.nan,
        } | {k: one[k] for k in t.columns if k.endswith("_ms_per_video_s")})
        pd.DataFrame(rows).to_csv(OUT / "trials.csv", index=False, float_format="%.4f", lineterminator="\n")
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False, float_format="%.4f", lineterminator="\n")
    import winreg  # shortcut: Windows only, as the laptop of D8

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    run = {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {
        "cpu": cpu, "logical_cores": os.cpu_count(), "ac_power_at_start": ac, "ac_power_at_end": on_ac_power(), "workers": WORKERS, "torch_threads_per_worker": 1,
        "seconds_per_trial": SECONDS, "subjects": subjects, "torch": torch.__version__, "opencv": cv2.__version__,
    }
    (OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n", newline="\n")
    print(f"Wrote trials.csv and summary.csv to {OUT}")


if __name__ == "__main__":
    main()
