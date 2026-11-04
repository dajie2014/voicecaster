"""验证 2026-11-04 最后一版分工：日常（讲课/朗读）小米 solo；演戏千问 + 方言 Vox。"""
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
print("小米:", server._omni_ready(), " 千问:", server._qwen_ready(), flush=True)


def t(tag, fn):
    t0 = time.time()
    w = fn()
    print(f"{tag:14s} {time.time()-t0:6.2f}s  音频 {len(w)/server.SR:5.2f}s", flush=True)


t("朗读(小米)", lambda: server._read_render({"text": "这是朗读测试。", "cid": card["id"]}, ""))
t("讲课(小米)", lambda: server._render_line(card, ref, "这是讲课测试。", "讲课", mode="lecture"))
t("演戏(千问)", lambda: server._render_line(card, ref, "这是演戏测试。", "演戏"))
dia = dict(card, dialect="四川话")
t("方言(Vox)", lambda: server._render_line(dia, ref, "这是方言测试。", "演戏"))
