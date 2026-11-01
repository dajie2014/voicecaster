"""把「动作/舞台提示」翻成「声音会变成什么样」。

只翻有人说话的提示；纯音效（叮、风扇声）返回空。
用本机 Ollama（qwen3:4b-instruct），一句约 0.4 秒。
"""
import json
import re
import urllib.request

URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:4b-instruct"

PROMPT = """你是广播剧的声音指导。下面是角色说台词时的动作、姿态或场面提示。

规则：
1. 只要提示里有角色（阿达、小迪、小奥、阿碧、旁白）在动作，就是"他一边做这个动作一边说这句话"，必须给出声音变化。
2. 只输出声音特征，逗号分隔，不超过 20 个字。比如：压低声音、说得慢、气息不稳、声音发闷、声音变远。
3. 只有完全没有角色、纯粹是音效或环境声（比如"叮的一声""风扇声变大"）时，才输出：无。

舞台提示：{text}
声音特征："""

# 只留「声音」这一类的词，动作词一律丢掉
_KEEP = re.compile(r"声音|语气|语速|气息|嗓音|音色|压低|变轻|变远|发闷|发抖|颤抖|沙哑|沉重|急促|缓慢|压低|带点|犹豫|忍着|叹息|喘")
_DROP = re.compile(r"手|脚|站|走|坐|转身|背对|抬头|低头|推|拉|放|停|指|敲|撑|扶|靠|眼|脸|头|肩")
# 纯音效/环境的提示：不涉及人声，别硬翻
_SFX = re.compile(r"叮|咚|噪音|风扇|嗡鸣|音效|环境|安静|空白|淡出|淡入|渐强|渐弱|机器声")
_ROLE = re.compile(r"阿达|小迪|小奥|阿碧|旁白")


def clean(raw):
    keep = []
    for part in re.split(r"[，,、；;]", raw):
        part = part.strip(" 。")
        if not part:
            continue
        if _DROP.search(part) and not _KEEP.search(part):
            continue
        if _KEEP.search(part):
            keep.append(part)
    return "，".join(keep[:3])


def to_sound(text, timeout=30):
    text = (text or "").strip()
    if not text:
        return ""
    if _SFX.search(text) and not _ROLE.search(text):
        return ""
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT.format(text=text),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 60},
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8")).get("response", "")
    except Exception as e:
        return ""
    out = out.strip().splitlines()[0].strip() if out.strip() else ""
    out = out.strip("。，, ")
    if not out or out in ("无", "無"):
        return ""
    return clean(out)


if __name__ == "__main__":
    tests = [
        "他站在办公室里，一只手撑着桌沿，另一只手指着屏幕，一动不动",
        "【叮——】账单弹窗弹出，机房风扇提速，噪音变大",
        "阿达站起身，走到窗前，背对着门，很久没有说话",
        "小奥把纸推过去，手停在半空，又缩回来",
    ]
    for t in tests:
        print(f"{t[:24]:26s} -> {to_sound(t) or '（不涉及人声）'}")
