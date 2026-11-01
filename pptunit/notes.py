"""从 pptx 里抽出每一页要讲的话。

规则（SPEC 第三节）：
- 备注优先；这一页没备注，就退到页面正文
- 正文按「从上到下、从左到右」排，别按形状创建顺序
- 占位提示语（"点击此处添加备注"）当没有

备注怎么写的规则在技能 pptx-lecture-notes：
~/.dsh/skills/pptx-lecture-notes/（讲课口气、只改备注、信息零丢失）
"""
import re
from pptx import Presentation
from pptx.util import Emu

PLACEHOLDER = {
    "点击此处添加备注", "单击此处添加备注", "Click to add notes",
    "click to add notes", "点击添加备注",
}
PAGE_NO_RE = re.compile(r"^\s*(\d{1,3}|第\s*\d+\s*页)\s*$")


def _text_of(shape):
    if shape.has_text_frame:
        return shape.text_frame.text.strip()
    if getattr(shape, "has_table", False) and shape.has_table:
        rows = []
        for row in shape.table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" ".join(c for c in cells if c))
        return "\n".join(rows)
    return ""


def _sorted_shapes(slide):
    def key(sh):
        return (sh.top if sh.top is not None else 0,
                sh.left if sh.left is not None else 0)
    return sorted(slide.shapes, key=key)


def body_text(slide):
    parts = []
    for sh in _sorted_shapes(slide):
        t = _text_of(sh)
        if not t or t in PLACEHOLDER:
            continue
        # 右下角孤零零一个页码，不是讲课内容
        if PAGE_NO_RE.match(t) and (sh.top or 0) > Emu(0) and t.isdigit():
            continue
        parts.append(t)
    return "\n".join(parts)


def notes_text(slide):
    if not slide.has_notes_slide:
        return ""
    t = slide.notes_slide.notes_text_frame.text.strip()
    if t in PLACEHOLDER:
        return ""
    return t


def extract(pptx_path):
    """返回 [{index, notes, body, source}]，source = 'notes' | 'body' | 'empty'"""
    prs = Presentation(pptx_path)
    pages = []
    for i, s in enumerate(prs.slides, 1):
        n = notes_text(s)
        b = body_text(s)
        src = "notes" if n else ("body" if b else "empty")
        pages.append({"index": i, "notes": n, "body": b, "source": src,
                      "text": n or b})
    return pages


if __name__ == "__main__":
    import sys
    pages = extract(sys.argv[1])
    print(f"{len(pages)} 页")
    for p in pages:
        print(f"\n=== 第 {p['index']} 页（{p['source']}）===")
        print((p["text"] or "（空）")[:200])
