"""量「讲课走小米」的稳态耗时（首句含热身，后面是真实速度）。"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import server  # noqa: E402

card = json.load(open("characters/1.json", encoding="utf-8"))
ref = os.path.join(HERE, "characters", card["ref"])
texts = [
    "这一讲我们讲声音是怎么被合成出来的。",
    "先看模型这一层，再看它怎么把文字变成波形。",
    "这一讲我们讲声音是怎么被合成出来的。",
]

print("小米就绪:", server._omni_ready(), flush=True)
for i, t in enumerate(texts, 1):
    t0 = time.time()
    wav = server._omni_wav(t, ref, card.get("ref_text") or "")
    print(f"第 {i} 句 {time.time() - t0:6.2f}s  音频 {len(wav) / server.SR:5.2f}s  字数 {len(t)}",
          flush=True)
