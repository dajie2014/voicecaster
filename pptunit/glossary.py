"""音乐术语表：中文讲解里夹的外语术语，要按那个语言的读法念。

VoxCPM2 支持 30 种语言、**不需要语言标签**，给什么语言的文本就读什么语言。
所以术语表干的事是：
1. 认出句子里的术语（整词匹配，长词优先）
2. 在术语和中文之间**加逗号断开**，让模型把它当独立的外语词读，
   而不是按中文语流吞掉（这是"汉字注音式误读"的根子）
3. 记录它属于哪种语言，导出配置时给人看

表是**用户自己维护的**（SPEC 硬约束：不许自动填第三方术语表）。
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "glossary.json")
CJK = r"\u4e00-\u9fff"


def load(path=DEFAULT_PATH):
    if not os.path.exists(path):
        return {"version": 1, "terms": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(data, path=DEFAULT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find(text, terms):
    """整词命中，长词优先。返回 [(词条, 起始位置)]，按出现顺序。"""
    hits, taken = [], []
    for t in sorted(terms, key=lambda x: -len(x["term"])):
        word = t["term"]
        pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", re.I)
        for m in pat.finditer(text):
            a, b = m.start(), m.end()
            if any(not (b <= x or a >= y) for x, y in taken):
                continue
            taken.append((a, b))
            hits.append((t, a, b))
    return sorted(hits, key=lambda x: x[1])


def _prev_non_space(s, i):
    j = i - 1
    while j >= 0 and s[j] in " \t\u3000":
        j -= 1
    return s[j] if j >= 0 else ""


def _next_non_space(s, i):
    j = i
    while j < len(s) and s[j] in " \t\u3000":
        j += 1
    return s[j] if j < len(s) else ""


def apply(text, terms):
    """在术语两侧断开（相邻是中文就加逗号），返回 (新文本, 命中的词)。

    前后本来有空格也算「紧贴中文」——否则「的 Allegro 要」这种写法，
    模型会把术语吞进中文语流里按中文读。
    """
    hits = find(text, terms)
    if not hits:
        return text, []
    # 从后往前改，避免位置错位
    out = text
    for t, a, b in reversed(hits):
        left_cjk = bool(re.match("[" + CJK + "]", _prev_non_space(out, a)))
        right_cjk = bool(re.match("[" + CJK + "]", _next_non_space(out, b)))
        seg = out[a:b]
        if left_cjk:
            out = out[:a] + "，" + seg + out[b:]
            b += 1
        if right_cjk:
            out = out[:b] + "，" + out[b:]
    # 中文逗号前后不该有空格（原文里「的 Allegro」这种会被插成「的 ，Allegro」）
    out = re.sub(r"\s*，\s*", "，", out)
    return out, [t["term"] for t, _, _ in hits]


if __name__ == "__main__":
    g = load()
    if not g["terms"]:
        raise SystemExit("术语表是空的：%s" % DEFAULT_PATH)
    s = "这一段的 Allegro 要弹得干脆，后面接 Piano，再回到 Forte。"
    new, hit = apply(s, g["terms"])
    print("原句：", s)
    print("处理后：", new)
    print("命中：", hit)
