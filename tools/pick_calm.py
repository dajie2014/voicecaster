"""从一段录音里挑出「最平的一段」，当克隆参考。

为什么（2026-10-30）：用户发现克隆音会「说着说着高亢」——
因为整段录音的情绪起伏都被学走了，生成时随机抽。
与其让他一次录准，不如让他录长一点，这里的量活儿交给程序：
算每一帧的基频（F0），在所有滑动窗口里挑抖动最小的一段。

用法：
    .venv/bin/python tools/pick_calm.py 录音.wav              # 挑 15 秒
    .venv/bin/python tools/pick_calm.py 录音.wav --len 12
    .venv/bin/python tools/pick_calm.py 录音.wav --out output/clone/picked_48k.wav
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = "/opt/homebrew/bin/ffmpeg"


def load(path):
    """统一转成 16k 单声道来量，任何格式都能进。"""
    if not path.lower().endswith(".wav"):
        tmp = "/tmp/_pick_in.wav"
        subprocess.run([FFMPEG, "-y", "-i", path, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", tmp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = tmp
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:                      # 量基频不需要高采样率
        y = y[::max(1, sr // 16000)].astype(np.float32)
        sr = 16000
    return y.astype(np.float32), sr


def f0_track(x, sr, fmin=60, fmax=400, win=0.04, hop=0.02):
    w, h = int(sr * win), int(sr * hop)
    out = []
    for i in range(0, len(x) - w, h):
        seg = x[i:i + w]
        if np.sqrt((seg ** 2).mean()) < 0.005:      # 太轻＝停顿/气声
            out.append(np.nan)
            continue
        seg = seg - seg.mean()
        c = np.correlate(seg, seg, "full")[w - 1:]
        lo, hi = int(sr / fmax), int(sr / fmin)
        if hi >= len(c):
            out.append(np.nan)
            continue
        k = lo + int(np.argmax(c[lo:hi]))
        out.append(sr / k if c[k] > 0.3 * c[0] else np.nan)
    return np.array(out), h / sr


def pick(path, want=15.0):
    y, sr = load(path)
    total = len(y) / sr
    f, dt = f0_track(y, sr)
    n = int(want / dt)
    if n >= len(f):
        v = f[~np.isnan(f)]
        return {"start": 0.0, "length": total, "median": float(np.median(v)) if len(v) else 0,
                "jitter": float(np.std(v)) if len(v) else 0, "total": total, "short": True}

    best = None
    for a in range(0, len(f) - n, 2):
        seg = f[a:a + n]
        v = seg[~np.isnan(seg)]
        if len(v) < n * 0.35:            # 有声音的帧太少（大段停顿）不算
            continue
        sd = float(np.std(v))
        if best is None or sd < best[0]:
            best = (sd, a, float(np.median(v)), len(v) / n)
    if best is None:
        return None
    sd, a, med, cover = best
    return {"start": round(a * dt, 2), "length": want, "median": round(med),
            "jitter": round(sd), "total": round(total, 1), "voiced": round(cover, 2),
            "short": False}


def cut(src, start, length, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run([FFMPEG, "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
                        "-i", src, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", out],
                       capture_output=True, text=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--len", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(BASE, "output", "clone", "picked_48k.wav"))
    a = ap.parse_args()

    r = pick(a.audio, a.len)
    if not r:
        print("量不出来：这段里几乎没有能用的说话声")
        return 1
    print(f"整段 {r['total']} 秒")
    if r["short"]:
        print("比要的短，整段都能用（建议录到 30 秒以上，好挑）")
    else:
        print(f"最平的一段：第 {r['start']} 秒起 {r['length']:.0f} 秒"
              f"（有声音的占 {r['voiced']*100:.0f}%）")
        print(f"  调门中位 {r['median']} Hz，起伏 {r['jitter']} Hz（越小越平）")
    if cut(a.audio, r["start"], r["length"], a.out):
        print("切好了：", a.out)
        return 0
    print("切不出来")
    return 1


if __name__ == "__main__":
    sys.exit(main())
