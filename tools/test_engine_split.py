"""引擎分工回归测试（2026-11-04）。

用法（在 voicecaster 目录、用 voxcpm 的 venv 跑）：
    python tools/test_engine_split.py
它会加载本机 Vox（约 14 秒），然后分别走：
  1) 演戏/朗读默认 → 千问
  2) 讲课 mode="lecture" → 小米 OmniVoice
  3) 带方言的卡 → Vox
只打印耗时与时长，不写项目文件。
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import server  # noqa: E402  （import 时就会加载 Vox）

TEXT = "这是一句用来检查引擎分工的话，念完就停。"


def run(tag, card, mode="perform"):
    t0 = time.time()
    # ★ 必须绝对路径：千问/小米是另一个进程，相对路径在它们那边找不到
    wav = server._render_line(card, os.path.join(HERE, "characters", card["ref"]), TEXT,
                              card.get("desc", "自然说话"), mode=mode)
    dt = time.time() - t0
    print(f"{tag:10s} 耗时 {dt:6.2f}s  时长 {len(wav) / server.SR:5.2f}s  "
          f"采样率 {server.SR}", flush=True)
    return dt


def main():
    card = json.load(open("characters/1.json", encoding="utf-8"))
    print("角色:", card.get("name"), "| engine =", card.get("engine"), flush=True)
    print("千问就绪:", server._qwen_ready(), " 小米就绪:", server._omni_ready(), flush=True)
    run("演戏/朗读", card, "perform")
    run("讲课（按卡）", card, "perform")     # 2026-11-04 晚：讲课改回跟角色卡走
    run("指定小米", card, "lecture")
    dia = dict(card, dialect="四川话")
    run("方言", dia, "perform")


if __name__ == "__main__":
    main()
