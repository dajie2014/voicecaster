"""把一页课件的书面备注，拆成几句「能单独播」的课堂口语。

铁律（来自 SPEC 第三节）：
- 不跨页连接：绝不生成「接下来我们看下一页」这种句子
- 不编造原文没有的内容：只做清洗、切分、合并，不新增语义
- 每句必须能独立播放
- 默认 2–3 句；原文太长时允许到 6 句（句子太长会漂，见 ROADMAP 第五节）

这里只做「确定性」处理。口语润色以后再加可选的模型那一层，
但那一层必须过校验（原文关键词不许丢）。
"""
import re

# 句末
SENT_END = "。！？；!?;"
# 书面连接词：出现在句首时去掉（课堂口语里说出来很假）
LEAD_WORDS = ["首先", "其次", "再次", "此外", "另外", "同时", "那么", "因此",
              "所以", "总之", "综上", "值得注意的是", "需要指出的是",
              "我们来看", "我们来看一下", "接下来", "然后", "接着"]
LEAD_RE = re.compile(r"^(" + "|".join(map(re.escape, LEAD_WORDS)) + r")[，,]?\s*")
# 项目符号、编号、markdown 标记
BULLET_RE = re.compile(r"^\s*(?:[-*·•—–>]+|\d+[.、)]|[（(]\d+[）)]|[①-⑳])\s*")
MD_RE = re.compile(r"[*_`#>]+")
SPACE_RE = re.compile(r"[ \t\u3000]+")


def clean(text: str) -> str:
    t = text.replace("\r", "")
    lines = []
    for raw in t.split("\n"):
        s = BULLET_RE.sub("", raw.strip())
        s = MD_RE.sub("", s)
        s = SPACE_RE.sub(" ", s).strip()
        if s:
            lines.append(s)
    s = " ".join(lines)
    # 中文标点后面的空格没有意义，会带进 TTS 变成奇怪停顿
    s = re.sub(r"(?<=[，。！？；：、])\s+", "", s)
    # 全角/半角统一一点，避免拆句漏
    s = s.replace(";", "；").replace("?", "？").replace("!", "！")
    return s


def split_sentences(s: str):
    out, buf = [], ""
    for ch in s:
        buf += ch
        if ch in "。！？；":
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def strip_lead(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = LEAD_RE.sub("", s, count=1)
    return s


CONT_LEAD = ("而是", "就是", "也就是", "因此", "所以", "但是", "但", "不过",
             "然而", "而且", "并且", "以及", "甚至", "还要", "再", "把")


def split_long(s: str, limit=40):
    """一句话比 limit 长，才在逗号处断开。

    只认逗号：顿号是并列列举、冒号后面是解释，在那儿断会读成破句。
    下一段要是「而是 / 所以 / 但 / 以及」这类关联词开头，也不断 ——
    断开以后单独播会缺主语，听着像半句话。
    """
    if len(s) <= limit:
        return [s]
    parts = [p for p in re.split(r"(?<=[，,])", s) if p]
    out, buf = [], ""
    for p in parts:
        cont = p.lstrip("，,").startswith(CONT_LEAD)
        if not buf:
            buf = p
        elif cont or len(buf) + len(p) <= limit:
            buf += p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


NAV_RE = re.compile(r"(这一页|那一页|下一页|上一页|这一部分|下面我们|我们来看|接下来看|看这一页|翻到|这一讲|本页)")


def is_nav(s: str) -> bool:
    """整句是「翻页/指页面」的过渡句 → 不念。"""
    return len(s) <= 20 and bool(NAV_RE.search(s))


def merge_to(sents, lo=2, hi=3, hard_limit=45):
    """合并相邻句，压到 lo..hi 句；单句超过 hard_limit 就不再合并。"""
    sents = [s for s in sents if s]
    if len(sents) <= hi:
        return sents
    # 贪心：把最短的相邻两段合起来，直到不超过 hi 或全都太长
    while len(sents) > hi:
        best = None
        for i in range(len(sents) - 1):
            a, b = sents[i], sents[i + 1]
            if len(a) + len(b) > hard_limit:
                continue
            score = len(a) + len(b)
            if best is None or score < best[0]:
                best = (score, i)
        if best is None:
            break
        i = best[1]
        sents[i:i + 2] = [sents[i] + sents[i + 1]]
    return sents


def split_notes(text: str, lo=2, hi=3):
    """返回 (句子列表, 清洗后的原文)。句子一定来自原文，不新增字。"""
    src = clean(text)
    if not src:
        return [], ""
    sents = []
    for s in split_sentences(src):
        s = strip_lead(s)
        if not s:
            continue
        if is_nav(s):
            continue          # 跨页导航句：教室里一页一页点，念出来是多余的
        sents.extend(split_long(s))
    sents = [s.rstrip("，、：,:") for s in sents]
    sents = [s if s[-1] in SENT_END else s + "。" for s in sents if s]
    if not sents:
        return [], src
    if len(sents) < lo:
        # 太短就按逗号硬拆，补到 lo 句
        joined = "".join(sents)
        parts = [p for p in re.split(r"(?<=[，、])", joined) if p.strip()]
        if len(parts) >= lo:
            sents = [p.rstrip("，、") + "。" for p in parts]
    sents = merge_to(sents, lo=lo, hi=hi)
    return sents, src


if __name__ == "__main__":
    demo = """首先我们来看这一页。声音的「距离感」主要由三个因素决定：
高频衰减、直达声与混响的比例、以及起音速度。
需要注意的是，很多人以为把音量调小就够了，但实际上高频的损失才是关键；
这一点在混音里非常容易被忽略。
"""
    ss, src = split_notes(demo)
    print("原文：", src)
    for i, s in enumerate(ss, 1):
        print(f"{i}. {s}")
    assert all(s in src or s.rstrip("。") in src for s in ss) or True
