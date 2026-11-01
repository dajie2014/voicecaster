"""端到端：一份 pptx → 每页拆成几句口语 → 一条 M4A → 嵌回幻灯片。

用法：
    python build.py 课件.pptx --out 课件_有声.pptx --char 1 --limit 3
    python build.py 课件.pptx --dry            # 只看拆出来的句子，不调模型

产出目录（默认 output/ppt/<文件名>/）：
    audio/001.m4a, 002.m4a ...   文件名 = 幻灯片号
    manifest.json                 每页几句、谁念、多长、命中哪些术语

铁律：只走 Backend A（VoxCPM2），永远不调 IndexTTS。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from pptx import Presentation
from pptx.util import Emu

import notes
import split
import glossary
import embed

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CHARS = os.path.join(BASE, "characters")
GAP = 0.35          # 句间停顿
PRE_ROLL = 0.35     # 页首留白，点下去不是硬起
CFG = 2.2
# PPT 讲解固定语气：自然讲课，不要朗诵腔
TONE = "语气自然、平稳，像在课堂上讲解，语速稍快"


def load_char(cid):
    p = os.path.join(CHARS, f"{cid}.json")
    if not os.path.exists(p):
        raise SystemExit(f"找不到角色卡 {cid}：{p}")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    ref = os.path.join(CHARS, c["ref"])
    if not os.path.exists(ref):
        raise SystemExit(f"角色卡 {cid} 的定妆音频不存在：{ref}")
    return c, ref


def to_m4a(wav_path, m4a_path):
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "128000",
                    wav_path, m4a_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def plan(pptx_path, terms, lo=2, hi=3):
    pages = notes.extract(pptx_path)
    out = []
    for p in pages:
        sents, src = split.split_notes(p["text"], lo=lo, hi=hi)
        fixed, hits = [], set()
        for s in sents:
            t, h = glossary.apply(s, terms)
            fixed.append(t)
            hits.update(h)
        out.append({**p, "sentences": fixed, "raw": src, "hits": sorted(hits)})
    return out


def build(pptx_path, out_path, char_id, steps, limit, dry):
    terms = glossary.load()["terms"]
    pages = plan(pptx_path, terms)
    todo = [p for p in pages if p["sentences"]]
    if limit:
        todo = todo[:limit]
    print(f"共 {len(pages)} 页，有讲解的 {len(todo)} 页")
    for p in todo:
        print(f"  第 {p['index']} 页（{p['source']}）", " | ".join(p["sentences"]))
    if dry:
        return

    name = os.path.splitext(os.path.basename(pptx_path))[0]
    outdir = os.path.join(BASE, "output", "ppt", name)
    adir = os.path.join(outdir, "audio")
    os.makedirs(adir, exist_ok=True)

    c, ref = load_char(char_id)
    from voxcpm import VoxCPM
    t0 = time.time()
    print("加载模型...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False,
                                   device="mps", optimize=False)
    sr = model.tts_model.sample_rate
    print("模型就绪 %.1fs" % (time.time() - t0), flush=True)

    icon = os.path.join(outdir, "play_icon.png")
    embed.make_icon(icon)

    prs = Presentation(pptx_path)
    slides = list(prs.slides)
    manifest = []
    for p in todo:
        idx = p["index"]
        t1 = time.time()
        chunks = [np.zeros(int(PRE_ROLL * sr), dtype=np.float32)]
        for s in p["sentences"]:
            wav = None
            for _try in range(2):
                try:
                    wav = model.generate(text=f"({TONE}){s}", reference_wav_path=ref,
                                         cfg_value=CFG, inference_timesteps=steps)
                    break
                except Exception as e:
                    print(f"  这一句出错重试：{e}", flush=True)
            if wav is None:
                print(f"  跳过一句：{s[:20]}", flush=True)
                continue
            chunks += [wav.astype(np.float32), np.zeros(int(GAP * sr), dtype=np.float32)]
        full = np.concatenate(chunks[:-1])
        wav_p = os.path.join(adir, f"{idx:03d}.wav")
        sf.write(wav_p, full, sr)
        m4a_p = os.path.join(adir, f"{idx:03d}.m4a")
        to_m4a(wav_p, m4a_p)
        os.remove(wav_p)
        embed.strip_audio(slides[idx - 1])       # 重做时先清旧按钮
        embed.embed_slide(slides[idx - 1], m4a_p, icon,
                          prs.slide_width, prs.slide_height, f"slide{idx:03d}")
        secs = len(full) / sr
        manifest.append({"slide": idx, "audio": os.path.basename(m4a_p),
                         "seconds": round(secs, 2), "sentences": p["sentences"],
                         "glossary": p["hits"], "source": p["source"]})
        print(f"第 {idx} 页完成 {secs:.1f}s（{time.time()-t1:.0f}s）", flush=True)

    prs.save(out_path)
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"pptx": os.path.basename(pptx_path), "char": c["id"],
                   "char_name": c.get("name", ""), "pages": manifest},
                  f, ensure_ascii=False, indent=2)
    print("已保存", out_path)
    print("音频包", adir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--char", type=int, default=1)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.pptx)[0] + "_有声.pptx"
    build(a.pptx, out, a.char, a.steps, a.limit, a.dry)
