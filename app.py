"""
VoiceCaster · 角色模块（最小可用版）
- 点「＋」用一句话描述捏一个角色
- 出候选 → 听 → 定下来
- 定下来的角色按创建顺序编号 1、2、3……
"""
import os, json, glob, time, random
import torch
import soundfile as sf
import gradio as gr
from voxcpm import VoxCPM
from parser import parse
from motion import to_sound

BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters")
os.makedirs(CHARS, exist_ok=True)

print("loading model...", flush=True)
MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
SR = MODEL.tts_model.sample_rate
print("model ready", flush=True)

# 定妆用的演示句：不挑剧情，只要够长、能把音色撑住（约 8-12 秒）
SAMPLE_TEXT = "今天的活儿，先把手上的事情理一理，别急着往我这儿推。"
# 每次捏都换一句，避免所有人拿同一句话定妆
SAMPLE_POOL = [
    "今天的活儿，先把手上的事情理一理，别急着往我这儿推。",
    "我这个人说话直，有什么不对的地方，你当场就可以指出来。",
    "你别站那么远，过来一点，把刚才那张单子再给我看一眼。",
    "这件事说起来也简单，就是一步一步来，中间别漏了哪一步。",
]


def cards():
    out = []
    for p in sorted(glob.glob(os.path.join(CHARS, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def save_card(desc, wav_path, text, name=""):
    n = 1
    existing = {c["id"] for c in cards()}
    while n in existing:
        n += 1
    card = {"id": n, "name": (name or "").strip(), "desc": desc,
            "ref": os.path.basename(wav_path), "text": text,
            "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(CHARS, f"{n}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return card


def char_choices():
    out = []
    for c in cards():
        nm = c.get("name") or "（没名字）"
        desc = (c.get("desc") or "")[:18]
        out.append(f"{c['id']} 号 · {nm}｜{desc}")
    return out


def delete_chars(selected):
    gone = []
    for s in (selected or []):
        try:
            n = int(s.split(" ")[0])
        except Exception:
            continue
        for ext in ("json", "wav"):
            p = os.path.join(CHARS, f"{n}.{ext}")
            if os.path.exists(p):
                os.remove(p)
        gone.append(str(n))
    msg = f"删掉了 {len(gone)} 个：{'、'.join(gone)} 号" if gone else "先勾上要删的角色。"
    return gr.update(choices=char_choices(), value=[]), msg


def render_board():
    cs = cards()
    if not cs:
        return "**还没有角色。** 点下面的 ＋ 捏第一个。"
    lines = []
    for c in cs:
        nm = c.get("name") or "（没名字）"
        lines.append(f"**{c['id']} 号 · {nm}**\n\n{c['desc']}")
    return "\n\n".join(lines)


def generate(desc, _unused=None):
    desc = (desc or "").strip()
    text = random.choice(SAMPLE_POOL)
    if not desc:
        return None, "先写一句描述：他是谁、多大年纪、声音什么样。", text
    torch.manual_seed(int(time.time() * 1000) % (2**31))
    wav = MODEL.generate(
        text=f"({desc}){text}",
        cfg_value=2.2, inference_timesteps=16,
    )
    p = os.path.join(CHARS, f"_cand_{int(time.time())}.wav")
    sf.write(p, wav, SR)
    return p, f"捏好了，先听听这一条。（{len(wav)/SR:.1f} 秒）", text


def confirm(desc, wav_path, text, name=""):
    if not wav_path or not os.path.exists(wav_path):
        return "还没有候选，先捏一条。", render_board()
    card = save_card(desc.strip(), wav_path, text, name)
    # 把候选转成正式角色音频
    final = os.path.join(CHARS, f"{card['id']}.wav")
    os.replace(wav_path, final)
    card["ref"] = os.path.basename(final)
    with open(os.path.join(CHARS, f"{card['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return f"定下来了 —— **{card['id']} 号 · {card.get('name') or '（没名字）'}**。", render_board()


def check_script(text):
    all_lines, warns = parse(text or "")
    lines = [x for x in all_lines if x.get("kind") == "line"]
    prompts = [x for x in all_lines if x.get("kind") == "prompt"]
    if not lines:
        return "**一句都没读出来。** 格式是每行一条「名字：台词」，冒号右边才是要念的内容。", []
    idmap = {c.get("name"): c["id"] for c in cards() if c.get("name")}
    names = {}
    for ln in lines:
        names[ln["name"]] = names.get(ln["name"], 0) + 1
    rows = ["| 编号 | 名字 | 句数 | 角色卡 |", "|---|---|---|---|"]
    for n, c in sorted(names.items(), key=lambda x: -x[1]):
        if n in idmap:
            rows.append(f"| **{idmap[n]} 号** | {n} | {c} | ✅ |")
        else:
            rows.append(f"| — | {n} | {c} | ❌ 还没捏这个角色 |")
    msg = f"**读出 {len(lines)} 句台词，{len(names)} 个角色。**\n\n" + "\n".join(rows)
    table = [[ln["no"], ln["name"], ln["text"], ""] for ln in lines]
    if prompts:
        msg += (f"\n\n另外认出 **{len(prompts)} 处动作/音效提示**（不念，只在排演时给你看）：\n\n"
                + "\n".join(f"- {p['text']}" for p in prompts[:12]))
    if warns:
        msg += f"\n\n有 {len(warns)} 处没读通：\n\n" + "\n".join(f"- {w}" for w in warns)
    return msg, table


OUTDIR = os.path.join(BASE, "output")


def attach_prompts(lines):
    """把动作/音效提示挂到它后面那句台词上。"""
    pend = []
    for ln in lines:
        if ln.get("kind") == "prompt":
            pend.append(ln["text"])
        else:
            ln["prompts"] = pend
            pend = []
    return lines


def rehearse(text, df=None):
    """整段跑完再出声：过程中只报进度，跑完把整条给出来。"""
    all_lines, _ = parse(text or "")
    all_lines = attach_prompts(all_lines)
    lines = [x for x in all_lines if x.get("kind") == "line"]
    if not lines:
        yield "先贴剧本（每行「名字：台词」）。", gr.update(visible=False), gr.update(visible=False)
        return
    idmap = {}
    for c in cards():
        if c.get("name"):
            idmap[c["name"]] = c
    miss = sorted({ln["name"] for ln in lines if ln["name"] not in idmap})
    if miss:
        yield ("这些名字还没有角色卡，先回角色页捏：" + "、".join(miss),
               gr.update(visible=False), gr.update(visible=False))
        return

    # 表格里手写的「说话声音」按句号覆盖
    manual = {}
    try:
        for row in (df or []):
            idx = int(row[0])
            val = str(row[3] or "").strip()
            if val:
                manual[idx] = val
    except Exception:
        pass

    os.makedirs(OUTDIR, exist_ok=True)
    outs = []
    for i, ln in enumerate(lines):
        c = idmap[ln["name"]]
        yield (f"排演中…… 第 {i + 1} / {len(lines)} 句 · {c['id']} 号 {ln['name']}",
               gr.update(visible=False), gr.update(visible=False))
        ctrl = c["desc"]
        if ln["emotion"]:
            ctrl = f"{ctrl}，{ln['emotion']}"
        if ln["no"] in manual:
            ctrl = f"{ctrl}，{manual[ln['no']]}"
        extra = []
        for pt in ln.get("prompts", []):
            snd = to_sound(pt)
            if snd:
                extra.append(snd)
        if extra:
            ctrl = f"{ctrl}，" + "，".join(extra)
            yield (f"排演中…… 第 {i + 1} / {len(lines)} 句 · {c['id']} 号 {ln['name']}"
                   f"（动作转成：{'，'.join(extra)}）",
                   gr.update(visible=False), gr.update(visible=False))
        wav = MODEL.generate(
            text=f"({ctrl}){ln['text']}",
            reference_wav_path=os.path.join(CHARS, c["ref"]),
            cfg_value=2.2, inference_timesteps=16,
        )
        p = os.path.join(OUTDIR, f"{i + 1:03d}_{c['id']}号_{ln['name']}.wav")
        sf.write(p, wav, SR)
        outs.append(p)

    import numpy as np
    chunks = []
    gap = np.zeros(int(0.4 * SR), dtype=np.float32)
    for p in outs:
        a, _ = sf.read(p, dtype="float32")
        chunks += [a, gap]
    full = np.concatenate(chunks[:-1])
    final = os.path.join(OUTDIR, "整段.wav")
    sf.write(final, full, SR)
    yield (f"排演完成：{len(lines)} 句，整段 {len(full) / SR / 60:.1f} 分钟。点旁边「试听」听整条。",
           gr.update(visible=True), gr.update(value=final, visible=True))


def play_all(path):
    if not path:
        return ""
    return (f'<audio controls autoplay style="width:100%" '
            f'src="/gradio_api/file={path}"></audio>')


with gr.Blocks(title="VoiceCaster") as demo:
    gr.Markdown("## VoiceCaster")
    with gr.Tabs():
        with gr.Tab("角色"):
            gr.Markdown("捏嗓子：写一句话描述，捏，听，定。定下来的按顺序编号。")
            char_box = gr.CheckboxGroup(choices=char_choices(), label="角色（勾上可以删）")
            with gr.Row():
                del_btn = gr.Button("删除勾选的", variant="stop", scale=0)
                refresh = gr.Button("刷新", scale=0, size="sm")
                del_msg = gr.Markdown("")
            board = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=3):
                    name = gr.Textbox(label="名字（剧本里就是这么写的）", placeholder="例：阿达")
                    desc = gr.Textbox(label="这个角色是什么声音", lines=2,
                                      placeholder="例：中年男人，说话干、有点累，中音，不笑")
                    with gr.Row():
                        btn = gr.Button("＋ 捏一个", variant="primary")
                        ok = gr.Button("就它了")
                    status = gr.Markdown("")
                    audio = gr.Audio(label="候选", type="filepath")
                    text_used = gr.State("")
            refresh.click(lambda: gr.update(choices=char_choices(), value=[]),
                          [], [char_box])
            del_btn.click(delete_chars, [char_box], [char_box, del_msg])
            btn.click(generate, [desc], [audio, status, text_used])
            ok.click(confirm, [desc, audio, text_used, name], [status, board])
            ok.click(lambda: gr.update(choices=char_choices(), value=[]), [], [char_box])

        with gr.Tab("剧本"):
            gr.Markdown("把剧本粘进来。**每行一条「名字：台词」**，冒号右边才是要念的内容。\n\n"
                        "- **整行括号包住的**（（动作）／【音效】／〔舞台提示〕）＝ 提示行，**显眼显示但绝不念**\n"
                        "- 台词中间夹的（情绪）也不念，只当语气参考")
            script_in = gr.Textbox(label="剧本", lines=14, placeholder="阿达：这次的任务，先核对截图信息。\n小迪：明白老板。")
            with gr.Row():
                chk = gr.Button("检查格式", variant="primary")
                reh = gr.Button("排演（整段跑完再出声）")
            chk_out = gr.Markdown("")
            tbl = gr.Dataframe(
                headers=["句", "角色", "台词", "说话声音（可空）"],
                datatype=["number", "str", "str", "str"],
                interactive=True, wrap=True, label="台词表 —— 想给哪句加语气，就在最后一格写")
            tip = gr.Markdown(
                "**这一列可以写什么**：\n"
                "- 轻：「声音压得很低，气声，贴着话筒」\n"
                "- 远：「声音变远，像从门外传来」\n"
                "- 慢：「说得很慢，一字一句」\n"
                "- 颤：「声音发抖，气息不稳」\n\n"
                "**别写**：怒吼、大喊、尖叫 —— 会崩，输出一串哼声。\n"
                "**也别写**：大声／小声 —— 它不听，音量留给后期。")
            prog = gr.Markdown("")
            with gr.Row():
                listen = gr.Button("▶ 试听整段", visible=False, variant="primary")
                player = gr.Audio(label="整段", type="filepath", visible=False)
            audio_html = gr.HTML("")
            chk.click(check_script, [script_in], [chk_out, tbl])
            reh.click(rehearse, [script_in, tbl], [prog, listen, player])
            listen.click(play_all, [player], [audio_html])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7863, show_error=True)
