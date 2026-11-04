"""外文判定 + 分发：中文走千问、外文走小米。"""
import json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import server

cases = [
    ("这一讲我们讲声音是怎么合成的。", False),
    ("这一讲我们讲 AI 合成的声音。", False),
    ("This lecture is about speech synthesis.", True),
    ("Diese Vorlesung handelt von Sprachsynthese.", True),
    ("Hello，这是我想使用的新的克隆声。", False),
]
for t, want in cases:
    got = server._looks_foreign(t)
    print(("✅" if got == want else "❌"), got, t[:40])

card = json.load(open("characters/1.json", encoding="utf-8"))
ref = os.path.join(HERE, "characters", card["ref"])
for tag, txt in [("中文讲课", "这一讲我们讲声音的合成。"),
                 ("英文讲课", "This lecture is about speech synthesis.")]:
    t0 = time.time()
    w = server._render_line(card, ref, txt, "讲课", mode="lecture")
    print(f"{tag:8s} {time.time()-t0:6.2f}s  音频 {len(w)/server.SR:5.2f}s")
