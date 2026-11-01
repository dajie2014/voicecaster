"""做一版「只含已完成页」的试听 PPT。

为什么要它（2026-10-30）：整本生成只在**最后**才存一次 pptx，
61 页要跑一个来小时，中途一个播放键都看不到。
这个脚本把已经生成好的那几页先嵌进去、另存一份，随时能点开听。

用法：
    .venv/bin/python tools/make_preview_pptx.py "AI音乐第六讲-草稿本"
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "pptunit"))

from pptx import Presentation          # noqa: E402
import embed as _pembed               # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not name:
        print("用法：make_preview_pptx.py <课件名>")
        return 2
    d = os.path.join(BASE, "output", "ppt", name)
    src = os.path.join(d, "源.pptx")
    if not os.path.exists(src):
        print("没有这份课件的源文件：", src)
        return 2

    icon = os.path.join(d, "play_icon.png")
    _pembed.make_icon(icon)
    adir = os.path.join(d, "audio")

    prs = Presentation(src)
    slides = list(prs.slides)
    n = 0
    for i, slide in enumerate(slides, start=1):
        hit = None
        for ext in ("mp3", "m4a", "wav"):
            p = os.path.join(adir, f"{i:03d}.{ext}")
            if os.path.exists(p):
                hit = p
                break
        if not hit:
            break                       # 后面的还没生成，就此打住
        _pembed.strip_audio(slide)      # 源课件里若已嵌过，先清掉，别叠两层
        _pembed.embed_slide(slide, hit, icon, prs.slide_width, prs.slide_height,
                            f"slide{i:03d}")
        n += 1

    if not n:
        print("一页都还没生成好，没东西可嵌")
        return 1

    out = os.path.join(d, f"试听版_已完成{n}页.pptx")
    prs.save(out)
    print(f"好了：{n} 页带播放键（共 {len(slides)} 页）")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
