"""VoiceCaster · 剧本解析器

格式就一条规矩：**每行一条「名字：台词」**，冒号右边才是要念的内容。
其余东西（分段线、【音效】、（情绪））都自动剥掉，不念。

用法：
    python3 parser.py <剧本文件> [--chars 角色名1,角色名2,...]
"""
import re
import sys
import json

# 名字后面跟的冒号。名字里可以带括号注释（「阿达（老板）」），
# 但整段都是括号的（「（动作）」）不算名字，下面会再挡一道。
NAME_RE = re.compile(r"^([^：:【】\[\]「」『』#>*|`]{1,24})[：:]\s*(.*)$")
# 音效：【...】 或 [...]
SFX_RE = re.compile(r"【[^】]*】|\[[^\]]*\]")
# 情绪：（...） 或 (...)
EMO_RE = re.compile(r"（[^）]*）|\([^)]*\)")
# 分段线：────── [01] 广告片头 ──────
SECTION_RE = re.compile(r"^[─—\-=]{3,}.*[─—\-=]{3,}$|^[─—\-=]{3,}")
# 整行被括号包住 = 提示行（动作 / 音效），显眼显示但绝不朗读
PROMPT_RE = re.compile(r"^[【（(〔\[](.+?)[】）)〕\]]\s*$")


def parse(text):
    """返回 (lines, warnings)。lines 里每项：
    {no, section, name, text, emotion, sfx}
    """
    lines, warnings = [], []
    section = ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if SECTION_RE.match(s):
            section = s.strip("─—-= ")
            continue
        # 整行括号包住 = 提示行（动作、音效），记下来但不念
        mp = PROMPT_RE.match(s)
        if mp:
            inner = mp.group(1).strip()
            if inner:
                lines.append({"no": None, "kind": "prompt", "text": inner})
            continue
        # Markdown 说明文字（标题、引用、表格、代码）
        if s[0] in "#>|`":
            continue
        m = NAME_RE.match(s)
        if not m:
            # 不是「名字：台词」的行 —— 要么是有人漏了名字，要么是说明文字
            warnings.append(f"这行读不出「名字：台词」，已跳过：{s[:40]}")
            continue
        name, body = m.group(1).strip(), m.group(2).strip()
        # 整段都是括号的（「（动作）：...」）不是人名，别收
        if not re.sub(r"[（(][^）)]*[）)]", "", name).strip():
            warnings.append(f"这行括号里像是动作提示，已跳过：{s[:40]}")
            continue
        if not body:
            warnings.append(f"「{name}」后面没有内容，已跳过")
            continue
        sfx = SFX_RE.findall(body)
        body = SFX_RE.sub("", body)
        emo = EMO_RE.findall(body)
        body = EMO_RE.sub("", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            warnings.append(f"「{name}」这句剥掉标记后是空的，已跳过")
            continue
        lines.append({
            "kind": "line",
            "no": sum(1 for x in lines if x.get("kind") == "line") + 1,
            "section": section,
            "name": name,
            "text": body,
            "emotion": "".join(e.strip("（）()") for e in emo).strip(),
            "sfx": [x.strip("【】[]") for x in sfx],
        })
    return lines, warnings


def report(lines, warnings, known=None):
    talk = [x for x in lines if x.get("kind") == "line"]
    prompts = [x for x in lines if x.get("kind") == "prompt"]
    names = {}
    for ln in talk:
        names[ln["name"]] = names.get(ln["name"], 0) + 1
    print(f"共 {len(talk)} 句台词，{len(names)} 个角色，{len(prompts)} 处动作/音效提示（不念）：")
    for n, c in sorted(names.items(), key=lambda x: -x[1]):
        flag = "" if (known is None or n in known) else "   ← 角色卡里没有这个名字"
        print(f"  {n}: {c} 句{flag}")
    if warnings:
        print(f"\n有 {len(warnings)} 处需要你看一眼：")
        for w in warnings:
            print("  -", w)
    return names


if __name__ == "__main__":
    path = sys.argv[1]
    known = None
    if "--chars" in sys.argv:
        known = set(sys.argv[sys.argv.index("--chars") + 1].split(","))
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines, warnings = parse(text)
    report(lines, warnings, known)
    if "--json" in sys.argv:
        print(json.dumps(lines, ensure_ascii=False, indent=2))
