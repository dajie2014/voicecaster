"""VoiceCaster · 自己写的后端（FastAPI）

- 角色：列表 / 试听 / 删除
- 剧本：解析 → 台词表（每句可单独写声音）→ 排演（异步，轮询进度）→ 整段试听
"""
import os, json, glob, time, uuid, threading, random, re, subprocess, gc
# demucs 的权重缓存放工作区（模型 shell 写不了 ~/.cache/torch）
os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "voxcpm-tentacle", ".torch"))
import torch, numpy as np, soundfile as sf
from fastapi import FastAPI, Request, Body
import base64, io
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from voxcpm import VoxCPM

from parser import parse
import dialects as _dial
from motion import to_sound

BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters")
OUTDIR = os.path.join(BASE, "output")
PROJDIR = os.path.join(BASE, "projects")
os.makedirs(PROJDIR, exist_ok=True)
WEB = os.path.join(BASE, "web")
os.makedirs(CHARS, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

SR = 48000            # VoxCPM2 的输出采样率（模型懒加载，所以写成常量）

TASKS = {}
LOCK = threading.RLock()      # RLock：_vox() 会在已持锁的路径里再取一次
PRE_ROLL = 1.0   # 分句文件前面的静音（秒）

# ── 大户（Vox）懒加载（2026-11-04 用户要的「内存保底」之一）──
#    它一装就是 12G 常驻，而现在主力是小户/中户，只有方言和捏嗓子用它。
_VOX = None
_VOX_LAST = 0.0
VOX_IDLE_SEC = int(os.environ.get("VC_VOX_IDLE", "600"))   # 闲置多久卸大户（测试可调小）

# ── 内存保底：最多同时两个「重」引擎（2026-11-04 用户要的）──
#    重 = 一次要吃 5G 以上。中户（小米）常驻只占 0.6G、生成时 4G，不算重。
#    ⚠️ 小户和中户是 launchd 常驻（KeepAlive），杀了会自己回来，所以**卸不掉**；
#       真能卸的只有大户（本进程）和戏骨（子进程）。满了先卸最久没用的。
MEM_CAP = 2
MEM_FLOOR_GB = 8.0            # 不管怎样给系统留这么多可用
ENGINE_GB = {"qwen": 6.1, "voxcpm": 12.0, "indextts": 24.0}
RESIDENT = {}                 # 引擎名 -> 最后使用时间
_RES_LOCK = threading.Lock()


def _mem_avail_gb():
    """macOS 的「可用内存」＝ free + inactive + speculative（活动监视器同口径）。"""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
        page = 16384
        vals = {}
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip().rstrip(".") or 0)
        return (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)
                + vals.get("Pages speculative", 0)) * page / 2 ** 30
    except Exception:
        return 0.0


def _vox_unload():
    global _VOX
    if _VOX is None:
        return
    with _RES_LOCK:
        _VOX = None
        RESIDENT.pop("voxcpm", None)
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    print("大户已卸载（腾出约 12G）", flush=True)


def _unload_engine(name):
    """能真卸的才卸；小户/中户是 launchd 常驻，杀了会自己回来，不算。"""
    if name == "voxcpm":
        _vox_unload()
        return True
    if name == "indextts":
        try:
            api_theater_stop()
            return True
        except Exception:
            return False
    return False


def _sync_resident():
    """小户在跑就记一笔（它是常驻，占名额但不参与淘汰）。"""
    try:
        if "qwen" not in RESIDENT and _qwen_ready():
            RESIDENT["qwen"] = time.time()
    except Exception:
        pass


def _admit(name):
    """要用 name 之前过这一关。返回 True = 可以加载。"""
    with _RES_LOCK:
        _sync_resident()
        RESIDENT[name] = time.time()
        if len(RESIDENT) > MEM_CAP:
            for _t, k in sorted((t, k) for k, t in RESIDENT.items() if k != name):
                if _unload_engine(k):
                    RESIDENT.pop(k, None)
                    if len(RESIDENT) <= MEM_CAP:
                        break
        need = ENGINE_GB.get(name, 0.0)
        if need and _mem_avail_gb() - need < MEM_FLOOR_GB:
            print(f"内存不够：要 {need:.0f}G，可用 {_mem_avail_gb():.1f}G，保底 {MEM_FLOOR_GB:.0f}G",
                  flush=True)
            return False
        return True


def _vox():
    """第一次用到才装 Vox（约 14 秒、12G）。闲置由 _vox_reaper 卸。"""
    global _VOX, _VOX_LAST
    if _VOX is None:
        if not _admit("voxcpm"):
            raise RuntimeError("内存不够，装不下大户：先关掉戏骨，或等别的活干完")
        print("loading Vox...", flush=True)
        _VOX = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False,
                                      device="mps", optimize=False)
        print("Vox ready", flush=True)
    _VOX_LAST = time.time()
    return _VOX


def _vox_reaper():
    """闲置超过 VOX_IDLE_SEC 就把大户卸掉。"""
    while True:
        time.sleep(int(os.environ.get("VC_REAP_EVERY", "60")))
        try:
            if _VOX is not None and time.time() - _VOX_LAST > VOX_IDLE_SEC:
                with LOCK:
                    if _VOX is not None and time.time() - _VOX_LAST > VOX_IDLE_SEC:
                        _vox_unload()
        except Exception:
            pass


threading.Thread(target=_vox_reaper, daemon=True).start()
print("Vox 懒加载：第一次用到才装（约 14 秒、12G；闲置 10 分钟自动卸）", flush=True)

QWEN = "http://127.0.0.1:3910"


def _qwen_ready():
    try:
        with _urlreq.urlopen(QWEN + "/health", timeout=3) as r:
            return bool(json.loads(r.read()).get("ready"))
    except Exception:
        return False


def _qwen_wav(text, ref=None, ref_text=None, voice=None, instruct=None,
              lang="Chinese", speed=1.0):
    """调千问常驻服务（3910），返回 numpy 波形（统一到本机 SR）。

    voice=官方音色（不克隆）／ref=克隆参考音（必须配 ref_text，千问的克隆模型缺它不干活）。
    """
    body = {"text": text, "lang": lang, "speed": float(speed)}
    if voice:
        body["voice"] = voice
    if ref and os.path.exists(ref):
        body["ref_audio"] = ref
        body["ref_text"] = ref_text or ""
    if instruct:
        body["instruct"] = instruct
    req = _urlreq.Request(QWEN + "/tts",
                          data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=1800) as r:
        data = r.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(np.asarray(wav, dtype="float32"), orig_sr=sr, target_sr=SR)
    return np.asarray(wav, dtype="float32")


OMNI = "http://127.0.0.1:3900"


def _omni_ready():
    try:
        with _urlreq.urlopen(OMNI + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _omni_wav(text, ref, ref_text="", instruct=None, lang="Chinese", speed=1.0):
    """调小米 OmniVoice（VoiceStudio 后端 3900）。

    ★ 2026-11-04 用户定的分工：讲课交给小米。
    直接把角色卡的定妆音频当参考音传过去，不用先在那边建档。
    OmniVoice 出 24k，统一重采样到本工程的 SR（48k）。
    """
    boundary = "----vc" + uuid.uuid4().hex
    body = io.BytesIO()

    def _part(name, value):
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(str(value).encode("utf-8"))
        body.write(b"\r\n")

    _part("text", text)
    _part("language", lang)
    _part("speed", speed)
    _part("num_step", 16)
    if ref_text:
        _part("ref_text", ref_text)
    if instruct:
        _part("instruct", instruct)
    with open(ref, "rb") as fh:
        raw = fh.read()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="ref_audio"; filename="ref.wav"\r\n')
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(raw)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = _urlreq.Request(OMNI + "/generate", data=body.getvalue(),
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _urlreq.urlopen(req, timeout=1800) as r:
        data = r.read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(np.asarray(wav, dtype="float32"), orig_sr=sr, target_sr=SR)
    return np.asarray(wav, dtype="float32")


_ASR = None


def _transcribe_ref(path):
    """参考音里说的那句原话（千问克隆必须同时给）。首次加载 whisper 约 3 秒。"""
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel
        _ASR = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2",
                            device="cpu", compute_type="int8")
    segs, _ = _ASR.transcribe(path, language="zh", vad_filter=True)
    return "".join(s.text for s in segs).strip()


def _save_card(card):
    p = os.path.join(CHARS, f"{card['id']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)


# ── 全局引擎（2026-11-04 用户要的：页面顶上一个下拉，一次换掉所有角色）──
GLOBAL_CFG = os.path.join(BASE, "webcfg.json")
_GLOBAL_ENGINES = ("qwen", "omnivoice", "voxcpm")


def _global_engine():
    """空串 / "follow" = 跟随角色卡；否则是强制的引擎名。"""
    try:
        with open(GLOBAL_CFG, encoding="utf-8") as f:
            return (json.load(f).get("engine") or "").strip()
    except Exception:
        return ""


_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")


def _looks_foreign(text):
    """是不是外文稿：汉字很少、拉丁字母很多。中文里夹几个英文词不算。"""
    t = text or ""
    cjk = len(_CJK.findall(t))
    lat = len(_LATIN.findall(t))
    return lat > 8 and cjk * 4 < lat


def _engine_for(card):
    """这个角色这一句到底用谁念：全局强制优先，其次角色卡，最后千问。"""
    g = _global_engine()
    if g in _GLOBAL_ENGINES:
        return g
    return (card.get("engine") or "qwen") if card else "qwen"


def _render_line(card, ref, text, ctrl, emotion="", man="", extra="", steps=16, mode="perform"):
    """按角色卡分发。

    ★ 2026-11-04 定的分工（当晚按用户的「不是千问最快吗」改过一版）：
      演戏、朗读、讲课 → 千问为主（卡上 engine 缺省就是 qwen）
      要小米：把那张角色卡的引擎改成「小米」，或调用方传 mode="lecture"（留给外文讲课）
      方言（演戏和朗读里的）→ 一律 Vox，这是它唯一还被指着用的地方
    """
    dia = (card.get("dialect") or "").strip()
    if dia:
        ctrl = (ctrl + "，用" + dia + "说") if ctrl else ("用" + dia + "说")
        with LOCK:
            return _vox().generate(text=f"({ctrl}){text}", reference_wav_path=ref,
                                  cfg_value=2.2, inference_timesteps=steps)

    eng = _engine_for(card)          # 全局下拉 > 角色卡 > 千问
    if eng == "voxcpm" and not ref:
        eng = "qwen"                 # 官方音色没有参考音，Vox 念不了
    if mode == "lecture":
        # ★ 2026-11-04 最终版：小米只接外文稿；中文讲课走千问，小米平时闲着
        eng = "omnivoice" if (ref and _looks_foreign(text)) else "qwen"

    if eng == "omnivoice" and ref:
        if _omni_ready():
            try:
                return _omni_wav(text, ref, card.get("ref_text") or "",
                                 instruct=emotion or None)
            except Exception:
                pass          # 小米抽风就退回千问，别让一整页讲课卡死
        eng = "qwen"

    if eng == "qwen":
        perform = "，".join([x for x in (emotion, man, extra) if x])
        if (card.get("kind") or "clone") == "official" and card.get("voice"):
            return _qwen_wav(text, voice=card["voice"], instruct=perform or None)
        if not card.get("ref_text"):
            rt = _transcribe_ref(ref)
            if rt:
                card["ref_text"] = rt
                try:
                    _save_card(card)
                except Exception:
                    pass
        # 2026-11-01：克隆这条路以前把语气指令丢掉了 —— 剧本里的（语音提示）对克隆声音全无效。
        return _qwen_wav(text, ref=ref, ref_text=card.get("ref_text") or "",
                         instruct=perform or None)
    with LOCK:
        return _vox().generate(text=f"({ctrl}){text}", reference_wav_path=ref,
                              cfg_value=2.2, inference_timesteps=steps)
app = FastAPI()


def cards():
    out = []
    for p in sorted(glob.glob(os.path.join(CHARS, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def attach_prompts(lines):
    pend = []
    for ln in lines:
        if ln.get("kind") == "prompt":
            pend.append(ln["text"])
        else:
            ln["prompts"] = pend
            pend = []
    return lines


# ---------- 项目：新建 / 保存 / 打开 ----------
class SaveIn(BaseModel):
    name: str
    script: str
    voices: dict = {}


@app.get("/api/projects")
def api_projects():
    out = []
    for f in sorted(glob.glob(os.path.join(PROJDIR, "*.json")), key=os.path.getmtime, reverse=True):
        out.append({"name": os.path.splitext(os.path.basename(f))[0],
                    "saved": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f)))})
    return out


@app.post("/api/project/save")
def api_save(inp: SaveIn):
    name = inp.name.strip() or "未命名"
    p = os.path.join(PROJDIR, f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"name": name, "script": inp.script, "voices": inp.voices,
                   "saved": time.strftime("%Y-%m-%d %H:%M")}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "name": name}


@app.get("/api/project/{name}")
def api_load(name: str):
    p = os.path.join(PROJDIR, f"{name}.json")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------- Word 导入 / 导出 ----------
class DocxIn(BaseModel):
    name: str = "剧本.docx"
    data: str      # base64


@app.post("/api/import/docx")
def api_import_docx(inp: DocxIn):
    from docx import Document
    raw = base64.b64decode(inp.data)
    doc = Document(io.BytesIO(raw))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for tb in doc.tables:
        for row in tb.rows:
            cells = [c.text.strip() for c in row.cells]
            if cells and any(cells):
                parts.append("　".join(cells))
    return {"text": "\n".join(parts)}


@app.post("/api/import/text")
def api_import_text(inp: DocxIn):
    """导入文稿/剧本：txt / md / docx / pdf 都读成纯文本。
    用户 2026-10-30：文件选项卡要能吃常见文本格式。"""
    raw = base64.b64decode(inp.data)
    ext = os.path.splitext(inp.name)[1].lower()
    text = ""
    try:
        if ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(raw))
            parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for tb in doc.tables:
                for row in tb.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append("　".join(cells))
            text = "\n".join(parts)
        elif ext == ".pdf":
            from pypdf import PdfReader
            rd = PdfReader(io.BytesIO(raw))
            text = "\n".join((pg.extract_text() or "").strip() for pg in rd.pages)
        elif ext == ".doc":
            from docx import Document
            doc = Document(io.BytesIO(raw))
            text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        else:
            text = ""
            for enc in ("utf-8", "gb18030", "utf-16"):
                try:
                    text = raw.decode(enc)
                    break
                except Exception:
                    continue
    except Exception as e:
        return JSONResponse({"error": "读不了这个文件：" + str(e)[:160]}, status_code=400)
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return JSONResponse({"error": "这个文件里读不出文字（可能是扫描件/图片）"}, status_code=400)
    return {"text": text, "name": inp.name}


class DocxOut(BaseModel):
    script: str
    voices: dict = {}
    title: str = "剧本"


@app.post("/api/export/docx")
def api_export_docx(inp: DocxOut):
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    lines = [x for x in attach_prompts(parse(inp.script)[0]) if x.get("kind") == "line"]
    doc = Document()
    h = doc.add_heading(inp.title, level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("导出时间：" + time.strftime("%Y-%m-%d %H:%M"))

    tb = doc.add_table(rows=1, cols=4)
    tb.style = "Table Grid"
    for i, t in enumerate(["句", "角色", "台词", "说话声音"]):
        tb.rows[0].cells[i].text = t
    for ln in lines:
        cells = tb.add_row().cells
        cells[0].text = str(ln["no"])
        cells[1].text = ln["name"]
        cells[2].text = ln["text"]
        cells[3].text = (inp.voices or {}).get(str(ln["no"]), "")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    from fastapi.responses import Response
    from urllib.parse import quote
    fname = quote(f"{inp.title}.docx")
    return Response(buf.read(),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fname}"})


@app.get("/api/characters")
def api_characters():
    return [{"id": c["id"], "name": c.get("name", ""), "desc": c.get("desc", ""),
             "engine": c.get("engine") or "qwen",
             "kind": c.get("kind") or "clone",
             "voice": c.get("voice") or "",
             "audio": f"/api/audio/characters/{c['ref']}" if c.get("ref") else ""} for c in cards()]


class SayIn(BaseModel):
    text: str = ""
    steps: int = 16
    emotion: str = ""      # 8 种情绪标签之一，空=角色本来的描述


@app.post("/api/characters/{cid}/say")
def api_char_say(cid: int, inp: SayIn):
    """用这个角色念一句任意话（试音色用，不落进项目）。"""
    p = os.path.join(CHARS, f"{cid}.json")
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个角色"}, status_code=404)
    with open(p, encoding="utf-8") as f:
        card = json.load(f)
    ref = os.path.join(CHARS, card["ref"]) if card.get("ref") else ""
    if ref and not os.path.exists(ref):
        return JSONResponse({"error": "这个角色的定妆音频不见了"}, status_code=400)
    text = (inp.text or "").strip() or "晚风掠过林间，我听见远处传来一声温柔的呼唤。"

    # ★ 2026-10-31 用户定：千问（qwen-tts 1.7B，3910）也可以念，先查在不在
    # ★ 2026-11-04：千问是演戏/朗读主力；小米（3900）也走同一条分发
    gen = _engine_for(card)          # 全局下拉 > 角色卡 > 千问
    if gen in ("qwen", "omnivoice"):
        if gen == "qwen" and not _qwen_ready():
            return JSONResponse({"error": "千问服务没起来（3910）；先拉 qwen-tts 服务"}, status_code=503)
        if gen == "omnivoice" and not _omni_ready():
            return JSONResponse({"error": "小米服务没起来（3900）；先拉 VoiceStudio 后端"}, status_code=503)
        try:
            wav = _render_line(card, ref, text, card.get("desc", ""), emotion=inp.emotion)
        except Exception as e:
            tag = "千问" if gen == "qwen" else "小米"
            return JSONResponse({"error": f"{tag}出错：{e}"}, status_code=503)
        tag = "千问" if gen == "qwen" else "小米"
        out = os.path.join(OUTDIR, f"试念_{cid}号_{tag}.wav")
        sf.write(out, wav, SR)
        return {"ok": True, "engine": gen,
                "url": "/api/audio/output/" + os.path.basename(out),
                "text": text, "seconds": round(len(wav) / SR, 2)}

    # ★ 2026-10-30 用户定：每个声音带「用谁念」的标记。
    #   戏骨（IndexTTS）对短素材更抓得住口气，但从网页抓来的怪声音也常只有几秒。
    if gen == "indextts":
        if not _theater_ready():
            try:
                api_theater_start()          # 现把它叫醒（要两分钟左右）
            except Exception:
                pass
            for _ in range(80):
                time.sleep(2)
                if _theater_ready():
                    break
            else:
                return JSONResponse({"error": "戏骨（IndexTTS）还没热好，一般要两分钟；等它好了再点一次"},
                                    status_code=503)
        body = json.dumps({"text": text, "ref": ref, "lang": "zh",
                           "emotion": "", "emotion_text": "", "emotion_ref": inp.emotion or "",
                           "emo_alpha": 0.8, "speed": 1.0}).encode("utf-8")
        req = _urlreq.Request(INDEXTTS + "/say", data=body,
                              headers={"Content-Type": "application/json"})
        try:
            with _urlreq.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
        except Exception as e:
            return JSONResponse({"error": f"戏骨出错：{e}"}, status_code=503)
        fn = d.get("file", "")
        return {"ok": True, "engine": "indextts",
                "url": "/api/audio/output/theater/" + fn,
                "text": text, "seconds": d.get("seconds", 0)}

    # 走到这里 = Vox。官方音色没参考音，退给千问（全局下拉强制 Vox 时也会碰到）
    if not ref:
        if not _qwen_ready():
            return JSONResponse({"error": "这条声音没参考音，Vox 念不了；千问服务也没起来（3910）"},
                                status_code=503)
        wav = _qwen_wav(text, voice=card.get("voice") or "vivian",
                        instruct=card.get("desc") or None)
        out = os.path.join(OUTDIR, f"试念_{cid}号_千问.wav")
        sf.write(out, wav, SR)
        return {"ok": True, "engine": "qwen",
                "url": "/api/audio/output/" + os.path.basename(out),
                "text": text, "seconds": round(len(wav) / SR, 2)}

    ctrl = card.get("desc", "") or "自然说话"
    if inp.emotion:
        ctrl += "，" + inp.emotion
    with LOCK:
        wav = _vox().generate(text=f"({ctrl}){text}", reference_wav_path=ref,
                             cfg_value=2.2, inference_timesteps=inp.steps)
    out = os.path.join(OUTDIR, f"试念_{cid}号.wav")
    sf.write(out, wav, SR)
    return {"ok": True, "url": "/api/audio/output/" + os.path.basename(out),
            "text": text, "seconds": round(len(wav) / SR, 2)}


class EngineIn(BaseModel):
    engine: str = "qwen"


@app.post("/api/characters/{cid}/engine")
def api_char_engine(cid: int, inp: EngineIn):
    """改这个声音用谁念：qwen（小户）/ omnivoice（中户）/ voxcpm（大户）/ indextts（戏骨）。"""
    p = os.path.join(CHARS, f"{cid}.json")
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个角色"}, status_code=404)
    with open(p, encoding="utf-8") as f:
        card = json.load(f)
    card["engine"] = inp.engine if inp.engine in ("qwen", "omnivoice", "voxcpm", "indextts") else "qwen"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"ok": True, "card": card}


class EngineGlobalIn(BaseModel):
    engine: str = "follow"


@app.get("/api/settings/engine")
def api_get_global_engine():
    """页面顶上那个全局下拉的当前值。"""
    return {"engine": _global_engine() or "follow"}


@app.post("/api/settings/engine")
def api_set_global_engine(inp: EngineGlobalIn):
    """follow = 跟随角色卡；qwen / omnivoice / voxcpm = 全部强制用它。"""
    e = inp.engine if inp.engine in ("follow",) + _GLOBAL_ENGINES else "follow"
    with open(GLOBAL_CFG, "w", encoding="utf-8") as f:
        json.dump({"engine": e}, f, ensure_ascii=False)
    return {"ok": True, "engine": e}


OFFICIAL_VOICES = [
    {"id": "vivian", "name": "薇薇安", "desc": "千问自带 · 女声，干净"},
    {"id": "serena", "name": "塞雷娜", "desc": "千问自带 · 女声"},
    {"id": "uncle_fu", "name": "付叔", "desc": "千问自带 · 中年男声"},
    {"id": "ryan", "name": "瑞恩", "desc": "千问自带 · 男声"},
    {"id": "aiden", "name": "艾登", "desc": "千问自带 · 男声"},
    {"id": "ono_anna", "name": "小野安娜", "desc": "千问自带 · 日语女声"},
    {"id": "sohee", "name": "素熙", "desc": "千问自带 · 韩语女声"},
    {"id": "eric", "name": "埃里克", "desc": "千问自带 · 四川话男声"},
    {"id": "dylan", "name": "迪兰", "desc": "千问自带 · 北京话男声"},
]


VOICEDIR = os.path.join(OUTDIR, "voices")
os.makedirs(VOICEDIR, exist_ok=True)
OFFICIAL_META = os.path.join(VOICEDIR, "official_meta.json")


def _official_meta():
    try:
        with open(OFFICIAL_META, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_official_meta(m):
    with open(OFFICIAL_META, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def _avatar_url(kind, vid):
    if kind == "official":
        p = os.path.join(VOICEDIR, "official_%s.png" % vid)
        return "/api/audio/output/voices/official_%s.png" % vid if os.path.exists(p) else ""
    p = os.path.join(CHARS, "%s.png" % vid)
    return "/api/audio/characters/%s.png" % vid if os.path.exists(p) else ""


@app.get("/api/voices/library")
def api_voice_library():
    """声库：官方（千问九条自带嗓子）＋ 用户（克隆 / 捏的 / 导入的角色卡）。

    2026-11-01：每条带 方言 / 头像 / 默认语气，声库页要能直接改这几样。
    """
    meta = _official_meta()
    official = []
    for v in OFFICIAL_VOICES:
        m = meta.get(v["id"], {})
        official.append({
            "kind": "official", "id": v["id"],
            "name": m.get("name") or v["name"],
            "desc": v["desc"],
            "dialect": m.get("dialect", ""),
            "avatar": _avatar_url("official", v["id"]),
            "engine": "qwen", "voice": v["id"], "audio": "",
        })
    users = []
    for c in cards():
        users.append({
            "kind": "char", "id": c["id"],
            "name": c.get("name", ""), "desc": c.get("desc", ""),
            "kindc": c.get("kind") or "clone",
            "engine": c.get("engine") or "qwen",
            "voice": c.get("voice") or "",
            "dialect": c.get("dialect", "") or "",
            "instruct": c.get("instruct", "") or "",
            "avatar": _avatar_url("char", c["id"]),
            "audio": f"/api/audio/characters/{c['ref']}" if c.get("ref") else "",
        })
    return {"official": official, "users": users}


class VoiceMetaIn(BaseModel):
    kind: str = "official"
    id: str = ""
    name: str | None = None
    dialect: str | None = None
    instruct: str | None = None


@app.post("/api/voices/meta")
def api_voice_meta(inp: VoiceMetaIn):
    """改一条声音的名字 / 方言 / 默认语气。"""
    if inp.kind == "official":
        m = _official_meta()
        e = m.setdefault(inp.id, {})
        if inp.name is not None:
            e["name"] = inp.name
        if inp.dialect is not None:
            e["dialect"] = inp.dialect
        _save_official_meta(m)
        return {"ok": True}
    p = os.path.join(CHARS, "%s.json" % inp.id)
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个声音"}, status_code=404)
    with open(p, encoding="utf-8") as f:
        card = json.load(f)
    if inp.name is not None:
        card["name"] = inp.name
    if inp.dialect is not None:
        card["dialect"] = inp.dialect
    if inp.instruct is not None:
        card["instruct"] = inp.instruct
    _save_card(card)
    return {"ok": True, "card": card}


class AvatarIn(BaseModel):
    kind: str = "char"
    id: str = ""
    data: str = ""      # base64（png / jpg）


@app.post("/api/voices/avatar")
def api_voice_avatar(inp: AvatarIn):
    """给一条声音配张照片（官方和用户的都行）。"""
    try:
        raw = base64.b64decode((inp.data or "").split(",")[-1])
    except Exception:
        return JSONResponse({"error": "这张图读不了"}, status_code=400)
    if not raw:
        return JSONResponse({"error": "这张图是空的"}, status_code=400)
    if inp.kind == "official":
        dst = os.path.join(VOICEDIR, "official_%s.png" % os.path.basename(inp.id))
    else:
        dst = os.path.join(CHARS, "%s.png" % os.path.basename(inp.id))
    with open(dst, "wb") as f:
        f.write(raw)
    return {"ok": True, "url": _avatar_url(inp.kind, inp.id) + "?t=" + str(int(time.time()))}


class VoicePreviewIn(BaseModel):
    voice: str = "vivian"
    text: str = ""


@app.post("/api/voices/preview")
def api_voice_preview(inp: VoicePreviewIn):
    if not _qwen_ready():
        return JSONResponse({"error": "千问服务没起来（3910）"}, status_code=503)
    text = (inp.text or "").strip() or "风从山上下来，带着松针的味道。"
    try:
        wav = _qwen_wav(text, voice=inp.voice)
    except Exception as e:
        return JSONResponse({"error": f"千问出错：{e}"}, status_code=503)
    p = os.path.join(OUTDIR, f"官方试听_{inp.voice}.wav")
    sf.write(p, wav, SR)
    return {"ok": True, "url": "/api/audio/output/" + os.path.basename(p),
            "text": text, "seconds": round(len(wav) / SR, 2)}


class VoiceImportIn(BaseModel):
    voice: str
    name: str = ""


@app.post("/api/voices/import")
def api_voice_import(inp: VoiceImportIn):
    """把一条官方音色收进声库（变成一张角色卡，之后剧本/PPT 都能选）。"""
    vo = next((v for v in OFFICIAL_VOICES if v["id"] == inp.voice), None)
    if not vo:
        return JSONResponse({"error": "没有这条官方音色"}, status_code=400)
    ids = [c["id"] for c in cards()]
    cid = (max(ids) if ids else 0) + 1
    card = {"id": cid, "name": (inp.name or vo["name"]).strip(), "desc": vo["desc"],
            "kind": "official", "voice": vo["id"], "engine": "qwen",
            "ref": "", "text": "", "created": time.strftime("%Y-%m-%d %H:%M")}
    _save_card(card)
    return {"ok": True, "card": card}


class RenameIn(BaseModel):
    name: str


@app.post("/api/characters/{cid}/rename")
def api_rename_char(cid: int, inp: RenameIn):
    p = os.path.join(CHARS, f"{cid}.json")
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个角色"}, status_code=404)
    with open(p, encoding="utf-8") as f:
        card = json.load(f)
    new = (inp.name or "").strip()
    if not new:
        return JSONResponse({"error": "名字不能是空的"}, status_code=400)
    card["name"] = new
    with open(p, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"ok": True, "card": card}


@app.post("/api/characters/{cid}/delete")
def api_del_char(cid: int):
    for ext in ("json", "wav", "png"):
        p = os.path.join(CHARS, f"{cid}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    return {"ok": True}


class ParseIn(BaseModel):
    text: str


import re as _re
import difflib as _difflib


def _norm_name(s):
    """角色名归一化：去掉括号里的说明（「阿达（老板）」→「阿达」）、空白。"""
    s = _re.sub(r"[（(][^）)]*[）)]", "", s or "")
    return _re.sub(r"\s+", "", s)


def _match_char(name, chars):
    """把剧本里的名字对到角色卡。返回 (char_id 或 None, 匹配方式)。"""
    n = _norm_name(name)
    if not n:
        return None, "none"
    # ★ 2026-10-30 用户定：剧本的唯一逻辑是对编号 ——
    # 写「1」「1号」「1 号」都直接对 1 号角色，名字随便叫什么。
    m = _re.match(r"^(\d{1,3})号?$", n)
    if m:
        cid = int(m.group(1))
        if any(c.get("id") == cid for c in chars):
            return cid, "number"
    exact = [c for c in chars if _norm_name(c.get("name", "")) == n]
    if exact:
        return exact[0]["id"], "exact"
    for c in chars:
        cn = _norm_name(c.get("name", ""))
        if cn and (cn in n or n in cn):
            return c["id"], "contains"
    best = (0.0, None)
    for c in chars:
        cn = _norm_name(c.get("name", ""))
        if not cn:
            continue
        r = _difflib.SequenceMatcher(None, n, cn).ratio()
        if r > best[0]:
            best = (r, c["id"])
    if best[0] >= 0.75:
        return best[1], "fuzzy"
    return None, "none"


@app.post("/api/parse")
def api_parse(inp: ParseIn):
    all_lines = attach_prompts(parse(inp.text)[0])
    lines = [x for x in all_lines if x.get("kind") == "line"]
    prompts = [x for x in all_lines if x.get("kind") == "prompt"]
    chars = cards()
    out_lines, summary = [], {}
    for ln in lines:
        cid, how = _match_char(ln["name"], chars)
        out_lines.append({"no": ln["no"], "name": ln["name"], "text": ln["text"],
                          "emotion": ln.get("emotion", ""),
                          "section": ln.get("section", ""),
                          "prompts": ln.get("prompts", []),
                          "char_id": cid, "match": how})
        s = summary.setdefault(ln["name"], {"name": ln["name"], "count": 0,
                                            "char_id": cid, "match": how})
        s["count"] += 1
    order = ["number", "exact", "contains", "fuzzy", "none"]
    return {
        "lines": out_lines,
        "prompts": [p["text"] for p in prompts],
        "summary": sorted(summary.values(), key=lambda x: (order.index(x["match"]), -x["count"])),
        "unmapped": sorted({x["name"] for x in summary.values() if x["char_id"] is None}),
    }


class RehearseIn(BaseModel):
    text: str
    voices: dict = {}
    section: str = ""      # 只排这一场；空 = 整本
    marks: dict = {}       # 2026-11-01：{句号: {spans:[...], pauses:[...]}}


def _line_wav(c, text, ctrl, spans, pauses, narr_tone=""):
    """剧本里的一行，按「选中的字各带一句语气 + 光标处插停顿」合成。

    2026-11-01 用户要的：把朗读那套（自然语言加语气、光标处插停顿）搬到演戏这边。
    复用朗读的 _read_segments / _read_render，标记为空就别走这条路。
    """
    blk = ReadBlock(text=text, cid=c["id"], instruct=ctrl or "", dialect="",
                    spans=list(spans or []), pauses=list(pauses or []))
    parts = []
    for sg in _read_segments(blk):
        if "pause" in sg:
            sec = max(0.0, min(30.0, float(sg["pause"])))
            parts.append(np.zeros(int(sec * SR), dtype=np.float32))
        else:
            w = _read_render(sg, "", narr_tone)
            if w is not None and len(w):
                parts.append(np.asarray(w, dtype=np.float32))
    if not parts:
        return None
    return np.concatenate(parts)


def run_rehearse(tid, text, voices, section="", marks=None):
    t = TASKS[tid]
    try:
        all_lines = attach_prompts(parse(text)[0])
        lines = [x for x in all_lines if x.get("kind") == "line"]
        if section:
            lines = [x for x in lines if x.get("section", "") == section]
            if not lines:
                t.update(state="error", msg=f"「{section}」里没有台词")
                return
        cards_all = cards()
        by_id = {c["id"]: c for c in cards_all}
        resolved = {}
        for ln in lines:
            cid, _how = _match_char(ln["name"], cards_all)
            resolved[ln["no"]] = cid
        miss = sorted({ln["name"] for ln in lines if resolved.get(ln["no"]) is None})
        if miss:
            t.update(state="error", msg="这些名字还没有角色卡：" + "、".join(miss))
            return
        outs, raws = [], []
        for i, ln in enumerate(lines):
            c = by_id[resolved[ln["no"]]]
            t.update(msg=f"第 {i + 1} / {len(lines)} 句 · {c['id']} 号 {c.get('name') or ln['name']}", progress=(i + 1) / len(lines))
            ctrl = c["desc"]
            if ln.get("emotion"):
                ctrl += "，" + ln["emotion"]
            man = (voices or {}).get(str(ln["no"]), "").strip()
            if man:
                ctrl += "，" + man
            extra = []
            for pt in ln.get("prompts", []):
                snd = to_sound(pt)
                if snd:
                    extra.append(snd)
            if extra:
                ctrl += "，" + "，".join(extra)
            # 2026-11-01：这一行里选中的字可以各带一句语气，光标处可以插停顿
            m = ((marks or {}).get(str(ln["no"])) or (marks or {}).get(ln["no"]) or {})
            spans = [ReadSpan(**x) for x in (m.get("spans") or [])]
            pauses = [ReadPause(**x) for x in (m.get("pauses") or [])]
            wav = None
            if spans or pauses:
                try:
                    wav = _line_wav(c, ln["text"], ctrl, spans, pauses)
                except Exception as e:
                    t.update(msg=f"第 {ln['no']} 句带标记出错，退回整句：{e}")
                    wav = None
            if wav is None:
                for _try in range(2):
                    try:
                        wav = _render_line(c, os.path.join(CHARS, c["ref"]), ln["text"], ctrl,
                                           emotion=ln.get("emotion", ""), man=man,
                                           extra="，".join(extra))
                        break
                    except Exception as e:
                        t.update(msg=f"第 {ln['no']} 句出错，重试：{e}")
            if wav is None:
                t.update(msg=f"第 {ln['no']} 句跳过（生成失败）")
                continue
            p = os.path.join(OUTDIR, f"{ln['no']:03d}_{c['id']}号_{c.get('name') or '角色'}.wav")
            # 分句文件前面留 1 秒静音：拖进 DAW 时起点是准的
            pad = np.zeros(int(PRE_ROLL * SR), dtype=wav.dtype)
            sf.write(p, np.concatenate([pad, wav]), SR)
            outs.append(p)
            raws.append(wav)          # 整段用它拼，不带前置静音
        chunks, gap = [], np.zeros(int(0.4 * SR), dtype=np.float32)
        # ★ 2026-11-03：记下每句在整段里的起点（送 Musio 时按这些点切片）
        starts, pos = [], 0.0
        for a in raws:
            starts.append(round(pos, 3))
            pos += len(a) / float(SR) + 0.4
            chunks += [a, gap]
        full = np.concatenate(chunks[:-1])
        final = os.path.join(OUTDIR, ("分场_" + _safe(section).replace(" ", "_") + ".wav") if section else "整段.wav")
        sf.write(final, full, SR)
        try:
            with open(final + ".sentences.json", "w", encoding="utf-8") as fh:
                json.dump({"starts": starts, "gap": 0.4, "sr": SR}, fh, ensure_ascii=False)
        except Exception:
            pass
        files = [{"name": os.path.basename(f),
                  "url": "/api/audio/output/" + os.path.basename(f)} for f in outs]
        t.update(state="done", msg=f"排演完成：{len(lines)} 句", progress=1.0,
                 audio="/api/audio/output/" + os.path.basename(final), files=files)
    except Exception as e:
        t.update(state="error", msg=f"出错：{e}")


@app.post("/api/rehearse")
def api_rehearse(inp: RehearseIn):
    tid = uuid.uuid4().hex[:8]
    TASKS[tid] = {"state": "running", "msg": "准备中", "progress": 0.0}
    threading.Thread(target=run_rehearse, args=(tid, inp.text, inp.voices, inp.section, inp.marks), daemon=True).start()
    return {"task": tid}


class LineIn(BaseModel):
    no: int
    name: str
    text: str
    emotion: str = ""
    voice: str = ""
    spans: list = []      # 2026-11-01：这一句里各片段的语气
    pauses: list = []     # 光标处插的停顿


@app.post("/api/rehearse/line")
def api_rehearse_line(inp: LineIn):
    """只重念这一句：文件名、前置静音都跟整段排演时一样，直接覆盖。"""
    c = next((x for x in cards() if x.get("name") == inp.name), None)
    if not c:
        return JSONResponse({"error": f"角色库里没有「{inp.name}」"}, status_code=400)
    ctrl = c.get("desc", "")
    if inp.emotion:
        ctrl += "，" + inp.emotion
    if inp.voice:
        ctrl += "，" + inp.voice
    wav = None
    if inp.spans or inp.pauses:
        try:
            wav = _line_wav(c, inp.text, ctrl,
                            [ReadSpan(**x) for x in inp.spans],
                            [ReadPause(**x) for x in inp.pauses])
        except Exception:
            wav = None
    if wav is None:
        wav = _render_line(c, os.path.join(CHARS, c["ref"]), inp.text, ctrl,
                           emotion=inp.emotion, man=inp.voice)
    pad = np.zeros(int(PRE_ROLL * SR), dtype=wav.dtype)
    name = f"{inp.no:03d}_{c['id']}号_{inp.name}.wav"
    p = os.path.join(OUTDIR, name)
    sf.write(p, np.concatenate([pad, wav]), SR)
    return {"ok": True, "file": name, "url": "/api/audio/output/" + name,
            "seconds": round(len(wav) / SR, 2)}


@app.get("/api/task/{tid}")
def api_task(tid: str):
    return TASKS.get(tid, {"state": "missing"})


@app.get("/api/audio/{rel:path}")
def api_audio(rel: str):
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


# ================= PPT 教学单元 =================
import subprocess as _sp
import sys as _sys
_sys.path.insert(0, os.path.join(BASE, "pptunit"))
import notes as _pnotes
import split as _psplit
import glossary as _pgloss
import embed as _pembed

PPTROOT = os.path.join(OUTDIR, "ppt")
os.makedirs(PPTROOT, exist_ok=True)
PPTTASKS = {}
PPT_GAP = 0.35
PPT_TONE = "语气自然、平稳，像在课堂上讲解，一句一句说清楚，语速不快"


def _safe(s):
    s = "".join(ch for ch in s if ch.isalnum() or ch in " _-()（）·.").strip(" .")
    return s or "未命名"


def _pdir(name):
    d = os.path.join(PPTROOT, _safe(name))
    os.makedirs(d, exist_ok=True)
    return d


def _ppt_plan(src):
    terms = _pgloss.load().get("terms", [])
    out = []
    for p in _pnotes.extract(src):
        sents, raw = _psplit.split_notes(p["text"])
        fixed, hits = [], set()
        for s in sents:
            t, h = _pgloss.apply(s, terms)
            fixed.append(t)
            hits.update(h)
        out.append({"index": p["index"], "source": p["source"],
                    "raw": raw, "sentences": fixed, "hits": sorted(hits)})
    return out


class PptIn(BaseModel):
    name: str
    data: str = ""


@app.post("/api/ppt/analyze")
def api_ppt_analyze(inp: PptIn):
    d = _pdir(inp.name)
    src = os.path.join(d, "源.pptx")
    if inp.data:
        raw = base64.b64decode(inp.data)
        # ★ 2026-10-30 踩到：导入「已经嵌过语音的成品」会把服务撑死
        #   （38MB 的 pptx 里塞着 61 段音频，解包就吃几个 G）。宁可拒绝，也别崩。
        if len(raw) > 25 * 1024 * 1024:
            return JSONResponse(
                {"error": "这份课件太大了（%.0fMB）。多半是「已经带语音的成品」——"
                          "那里面嵌着几十段音频，读进来会把程序撑爆。请导入原来那份原稿。" % (len(raw) / 1048576)},
                status_code=400)
        with open(src, "wb") as f:
            f.write(raw)
    elif not os.path.exists(src):
        return JSONResponse({"error": "没有课件"}, status_code=400)
    # 已经生成过的页，把音频地址一起带回去，网页里就能直接试听
    existing = {}
    mp = os.path.join(d, "manifest.json")
    if os.path.exists(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                data = json.load(f)
            for e in data.get("pages", []):
                existing[str(e["slide"])] = {
                    "url": f"/api/audio/output/ppt/{os.path.basename(d)}/audio/{e['audio']}",
                    "seconds": e.get("seconds", 0)}
        except Exception:
            pass
    return {"name": os.path.basename(d), "pages": _ppt_plan(src), "existing": existing}


class PptBuildIn(BaseModel):
    name: str
    char: str = "1"      # 2026-11-01：也能是官方音色的 "o:vivian"
    steps: int = 32
    only: list = []
    pages: list = []      # 页面上改过的句子 [{index, sentences}]，有就盖掉自动拆的
    skip_existing: bool = False   # 只补还没生成音频的页
    fmt: str = "mp3"             # 输出音频格式：mp3（默认，老电脑也能放）/ m4a / wav
    speed: float = 1.0           # 语速倍数（<1 更慢，音高不变）。2026-10-30 用户听过对比：原速最好


def _ensure_buttons(prs, adir, icon):
    """把「已经有音频的页」都补上播放按钮，只补缺的。

    为什么要它（2026-10-30）：只重做某一页时，prs 是从源课件重新打开的，
    别的页虽然有音频，但按钮不在里面 —— 直接存盘会把整本按钮弄丢。
    """
    for i, slide in enumerate(prs.slides, start=1):
        if any((sh.name or "").startswith("audio_") for sh in slide.shapes):
            continue
        hit = None
        for ext in ("mp3", "m4a", "wav"):
            p = os.path.join(adir, f"{i:03d}.{ext}")
            if os.path.exists(p):
                hit = p
                break
        if not hit:
            continue
        try:
            _pembed.embed_slide(slide, hit, icon, prs.slide_width, prs.slide_height,
                                f"slide{i:03d}")
        except Exception:
            pass


def _save_pptx_quiet(prs, path, adir=None, icon=None):
    """中途落盘。

    2026-10-30 用户实测踩到：整本 61 页要跑一个来小时，而 pptx 只在最后存一次，
    中途打开看到的是一个播放键都没有。改成每页存一次（失败也不打断生成）。
    """
    try:
        if adir and icon:
            _ensure_buttons(prs, adir, icon)
        prs.save(path)
    except Exception:
        pass                            # 文件被 PowerPoint 打开占着时就这样，无妨


def _run_ppt(tid, inp):
    t = PPTTASKS[tid]
    try:
        d = _pdir(inp.name)
        src = os.path.join(d, "源.pptx")
        pages = _ppt_plan(src)
        # 页面上手动改过的句子优先（拆句只是初稿，讲课的人最有发言权）
        if inp.pages:
            edited = {p.get("index"): p.get("sentences") for p in inp.pages}
            for p in pages:
                s = edited.get(p["index"])
                if s:
                    p["sentences"] = [x.strip() for x in s if x and x.strip()]
        todo = [p for p in pages if p["sentences"]
                and (not inp.only or p["index"] in inp.only)]
        if inp.skip_existing:
            adir0 = os.path.join(d, "audio")
            todo = [p for p in todo
                    if not os.path.exists(os.path.join(adir0, f"{p['index']:03d}.m4a"))]
        # 2026-11-01：这里以前只认数字编号；现在官方音色用 "o:<voice>" 也能挑，
        #   统一走 _read_card，它两种都认。
        c = _read_card(inp.char)
        if not c:
            t.update(state="error", msg=f"没有 {inp.char} 号角色")
            return
        ref = os.path.join(CHARS, c["ref"]) if c.get("ref") else ""
        from pptx import Presentation
        prs = Presentation(src)
        slides = list(prs.slides)
        icon = os.path.join(d, "play_icon.png")
        _pembed.make_icon(icon)
        adir = os.path.join(d, "audio")
        os.makedirs(adir, exist_ok=True)
        total = sum(len(p["sentences"]) for p in todo) or 1
        done = 0
        manifest = []
        # 输出名提到循环前：每做完一页就落盘一次，随手能打开听
        out = os.path.join(d, os.path.basename(d) + "_有声.pptx")
        for p in todo:
            i = p["index"]
            chunks = [np.zeros(int(0.35 * SR), dtype=np.float32)]
            for s in p["sentences"]:
                wav = None
                for _try in range(2):        # 一句崩了重试一次，别拖垮整册
                    try:
                        # ★ 2026-11-04 用户定的最后一版：「一般人类情况，小米 solo」——讲课固定走中户（小米）
                        wav = _render_line(c, ref, s, PPT_TONE, steps=inp.steps, mode="lecture")
                        break
                    except Exception as e:
                        t.update(msg=f"第 {i} 页有一句出错，重试：{e}")
                if wav is None:
                    t.update(msg=f"第 {i} 页跳过一句：{s[:20]}")
                    done += 1
                    continue
                chunks += [wav.astype(np.float32), np.zeros(int(PPT_GAP * SR), dtype=np.float32)]
                done += 1
                t.update(msg=f"第 {i} 页 · 第 {done}/{total} 句", progress=done / total)
            full = np.concatenate(chunks[:-1])
            wav_p = os.path.join(adir, f"{i:03d}.wav")
            sf.write(wav_p, full, SR)
            fmt = inp.fmt if inp.fmt in ("m4a", "mp3", "wav") else "mp3"
            # 语速：整页做完再统一变速（atempo 保持音高）。2026-10-30 用户嫌快。
            sp = float(getattr(inp, "speed", 1.0) or 1.0)
            af = ["-af", f"atempo={sp:.3f}"] if abs(sp - 1.0) > 0.005 else []
            if fmt == "wav":
                final = wav_p
                if af:
                    slow_p = wav_p.replace(".wav", "_sp.wav")
                    _sp.run([FFMPEG, "-y", "-i", wav_p, *af,
                             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", slow_p],
                            check=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                    os.replace(slow_p, wav_p)
            else:
                final = os.path.join(adir, f"{i:03d}.{fmt}")
                _sp.run([FFMPEG, "-y", "-i", wav_p, *af, *_ff_args(fmt), final],
                        check=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                if os.path.exists(wav_p):
                    os.remove(wav_p)
            # 换了格式就把这一页别的格式的旧文件清掉，别让音频包里混两份
            for ext in ("m4a", "mp3", "wav"):
                old = os.path.join(adir, f"{i:03d}.{ext}")
                if old != final and os.path.exists(old):
                    os.remove(old)
            _pembed.strip_audio(slides[i - 1])      # 重做这一页时先清掉旧按钮
            _pembed.embed_slide(slides[i - 1], final, icon, prs.slide_width, prs.slide_height,
                                f"slide{i:03d}")
            manifest.append({"slide": i, "audio": os.path.basename(final),
                             "format": fmt,
                             "seconds": round(len(full) / SR / sp, 2),
                             "sentences": p["sentences"], "glossary": p["hits"],
                             "source": p["source"]})
            _save_pptx_quiet(prs, out, adir, icon)   # 做完一页就先存，别让人干等一小时
        _save_pptx_quiet(prs, out, adir, icon)
        mp = os.path.join(d, "manifest.json")
        merged = {}
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    for e in json.load(f).get("pages", []):
                        merged[e["slide"]] = e
            except Exception:
                merged = {}
        for e in manifest:                    # 这一轮做的盖掉旧的，别人的留着
            merged[e["slide"]] = e
        with open(mp, "w", encoding="utf-8") as f:
            json.dump({"pptx": os.path.basename(src), "char": c["id"],
                       "char_name": c.get("name", ""),
                       "pages": [merged[k] for k in sorted(merged)]},
                      f, ensure_ascii=False, indent=2)
        t.update(state="done", progress=1.0, msg=f"好了：{len(manifest)} 页，{total} 句")
    except Exception as e:
        t.update(state="error", msg=f"出错：{e}")


@app.get("/api/ppt/list")
def api_ppt_list():
    """已经导入过的课件都在这儿。

    2026-10-30 踩到：刷新页面后只能靠「重新上传文件」，而文件名一变就成了新课件、
    前面的音频全认不回来。这个列表让页面直接按名字装回来。
    """
    out = []
    root = os.path.join(OUTDIR, "ppt")
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "源.pptx")):
            continue
        adir = os.path.join(d, "audio")
        n_audio = len(glob.glob(os.path.join(adir, "*"))) if os.path.isdir(adir) else 0
        pages = 0
        mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    pages = len(json.load(f).get("pages", []))
            except Exception:
                pages = 0
        out.append({"name": name, "audio": n_audio, "done_pages": pages})
    out.sort(key=lambda x: -x["done_pages"])
    return out


@app.post("/api/ppt/build")
def api_ppt_build(inp: PptBuildIn):
    d = _pdir(inp.name)
    if not os.path.exists(os.path.join(d, "源.pptx")):
        return JSONResponse({"error": "没有这份课件，先导入一次"}, status_code=400)
    tid = uuid.uuid4().hex[:8]
    PPTTASKS[tid] = {"state": "running", "msg": "准备中", "progress": 0.0}
    threading.Thread(target=_run_ppt, args=(tid, inp), daemon=True).start()
    return {"task": tid}


class CleanupIn(BaseModel):
    name: str


@app.post("/api/ppt/cleanup")
def api_ppt_cleanup(inp: CleanupIn):
    """写回原课件之后，把工作区里的中间产物清掉。

    只删可以再生的东西（音频、_有声、试听版、manifest），**源.pptx 永远留着** ——
    留着它，下次想重做一页不用重新导入一遍。
    """
    d = _pdir(inp.name)
    if not os.path.isdir(d):
        return JSONResponse({"error": "没有这份课件"}, status_code=400)
    n = 0
    adir = os.path.join(d, "audio")
    if os.path.isdir(adir):
        for f in glob.glob(os.path.join(adir, "*")):
            try:
                os.remove(f)
                n += 1
            except Exception:
                pass
        try:
            os.rmdir(adir)
        except Exception:
            pass
    for pat in ("*_有声.pptx", "试听版_*.pptx", "manifest.json"):
        for f in glob.glob(os.path.join(d, pat)):
            try:
                os.remove(f)
                n += 1
            except Exception:
                pass
    return {"ok": True, "removed": n}


@app.get("/api/ppt/task/{tid}")
def api_ppt_task(tid: str):
    return PPTTASKS.get(tid, {"state": "missing"})


@app.get("/api/ppt/download/{name}")
def api_ppt_download(name: str):
    d = _pdir(name)
    files = sorted(glob.glob(os.path.join(d, "*_有声.pptx")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        return JSONResponse({"error": "还没生成"}, status_code=404)
    from urllib.parse import quote
    # 用户 2026-10-30 定：下载叫原课件的名字（不带「_有声」），拖回原文件夹时
    # 系统直接问「是否替换」，一步到位。磁盘上仍存 `_有声`，不会自我覆盖。
    fn = quote(os.path.basename(d) + ".pptx")
    return FileResponse(files[0],
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fn}"})


@app.get("/api/ppt/glossary")
def api_ppt_glossary():
    return _pgloss.load()


class GlossIn(BaseModel):
    terms: list


@app.post("/api/ppt/glossary")
def api_ppt_glossary_set(inp: GlossIn):
    data = {"version": 1, "terms": inp.terms}
    _pgloss.save(data)
    return {"ok": True, "count": len(inp.terms)}


# ================= 声音克隆（讲课要用自己的声音） =================
import shutil as _sh

_FFMPEG = "/opt/homebrew/bin/ffmpeg"
CLONEDIR = os.path.join(OUTDIR, "clone")
os.makedirs(CLONEDIR, exist_ok=True)


def _clean_clone_cache(days=1):
    """试听/候选只是临时缓存：启动时把超过一天没动过的清掉，别留残渣。"""
    now = time.time()
    for f in glob.glob(os.path.join(CLONEDIR, "*")):
        try:
            if os.path.isfile(f) and now - os.path.getmtime(f) > days * 86400:
                os.remove(f)
        except Exception:
            pass


_clean_clone_cache()
CLONE_TEXT = "各位同学，今天我们讲这一课。"
CLONE_TONE = "自然、平稳，像在课堂上讲课，语速稍快"


def _clone_path(name):
    return os.path.join(CLONEDIR, os.path.basename(name))


def _denoise(src, dst, strength=0.8):
    """轻降噪：压掉空调、风扇、电流那种**稳定**底噪，不碰人声。

    2026-11-01 加。用 noisereduce 的谱减（stationary=True）：
    它拿整段估计噪声谱，再按比例从每个频点减掉。
    strength 是"减多少"，0.8 = 减八成 —— 留一点，免得人声发干、齿音被削秃。
    """
    import noisereduce as _nr
    data, sr = sf.read(src, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 噪声样本要「自己找最安静的那 0.4 秒」——
    # 2026-11-01 实测：不给样本时（默认全段估计），人声说话段会被削掉 40%；
    # 给了安静样本，底噪照样降 13 dB，人声几乎不动（0.033 → 0.030）。
    y_noise = None
    win = int(0.4 * sr)
    if len(data) > win * 2:
        step = max(1, win // 4)
        best_i, best_rms = 0, None
        for i in range(0, len(data) - win + 1, step):
            r = float(np.sqrt(np.mean(np.square(data[i:i + win]))))
            if best_rms is None or r < best_rms:
                best_rms, best_i = r, i
        y_noise = data[best_i:best_i + win]
    out = _nr.reduce_noise(y=data, sr=sr, stationary=True,
                           prop_decrease=float(strength), n_fft=1024,
                           n_jobs=1, y_noise=y_noise)
    sf.write(dst, out, sr)
    return dst


class CloneIn(BaseModel):
    filename: str = "voice.wav"
    data: str
    denoise: bool = True


@app.post("/api/clone/upload")
def api_clone_upload(inp: CloneIn):
    """收一段录音，统一转成 48k 单声道 wav，并报时长。
    2026-11-01：顺手过一遍轻降噪（原始也留着，能来回听）。"""
    raw = base64.b64decode(inp.data)
    ext = os.path.splitext(inp.filename)[1].lower() or ".wav"
    src = os.path.join(CLONEDIR, "upload_raw" + ext)
    with open(src, "wb") as f:
        f.write(raw)
    dst = os.path.join(CLONEDIR, "upload_48k.wav")
    if os.path.exists(dst):
        os.remove(dst)
    # 先用 ffmpeg：网页录音是 webm/opus，afconvert 不认
    r = _sp.run([_FFMPEG, "-y", "-i", src, "-ar", "48000", "-ac", "1",
                 "-c:a", "pcm_s16le", dst], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        r = _sp.run(["afconvert", "-f", "WAVE", "-d", "LEI16@48000", "-c", "1", src, dst],
                    capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        return JSONResponse({"error": "这个格式读不了，换 wav/mp3/m4a 试试",
                             "detail": (r.stderr or "")[:300]}, status_code=400)
    d, sr = sf.read(dst)
    secs = round(len(d) / sr, 2)
    ref = os.path.basename(dst)
    url = "/api/audio/output/clone/upload_48k.wav"
    url_raw = url
    denoised = False
    if inp.denoise and secs >= 1.0:
        try:
            dn = os.path.join(CLONEDIR, "upload_dn.wav")
            _denoise(dst, dn)
            ref = "upload_dn.wav"
            url = "/api/audio/output/clone/upload_dn.wav"
            denoised = True
        except Exception as e:  # noqa: BLE001
            print("denoise failed: %s" % e, flush=True)
    return {"ref": ref, "seconds": secs, "url": url, "url_raw": url_raw,
            "denoised": denoised,
            "ok": 5 <= secs <= 60,
            "hint": "" if 5 <= secs <= 60 else
                    ("太短了，至少要 5 秒才能抓住音色" if secs < 5 else
                     "太长了，15 秒左右最好；太长反而容易带进别的环境声")}


class ClonePreviewIn(BaseModel):
    ref: str
    text: str = ""
    dialect: str = ""        # 方言：东北话 / 四川话 / 粤语 …（空＝普通话）


class DialectIn(BaseModel):
    text: str = ""
    dialect: str = ""


@app.post("/api/dialect/convert")
def api_dialect_convert(inp: DialectIn):
    """把普通话换成方言用字（只建议，转完还能自己改）。"""
    out, n, hits = _dial.convert(inp.text, inp.dialect)
    return {"text": out, "count": n, "hits": hits, "dialect": inp.dialect}


@app.post("/api/dialect/all")
def api_dialect_all(inp: DialectIn):
    """一次出九种（纯词表替换，离线、快，不改意思）。"""
    return {"items": _dial.convert_all(inp.text)}


@app.post("/api/clone/preview")
def api_clone_preview(inp: ClonePreviewIn):
    """用这段参考念一句标准话，让人确认「像不像我」。"""
    ref = _clone_path(inp.ref)
    if not os.path.exists(ref):
        return JSONResponse({"error": "没有这份参考音频，先上传"}, status_code=400)
    text = (inp.text or "").strip() or CLONE_TEXT
    # 和 PPT 用同一套术语断句，否则拿术语来试听会读得不一样
    try:
        text, hits = _pgloss.apply(text, _pgloss.load().get("terms", []))
    except Exception:
        hits = []
    tone = CLONE_TONE
    if inp.dialect:
        # 2026-10-30：VoxCPM2 有方言底子，但**靠方言书面字最准**；
        # 这里再把方言名写进语气描述里推一把（普通话文本也能有点味儿）。
        tone = f"{CLONE_TONE}，用{inp.dialect}说"
    with LOCK:
        wav = _vox().generate(text=f"({tone}){text}", reference_wav_path=ref,
                             cfg_value=2.2, inference_timesteps=32)
    p = os.path.join(CLONEDIR, "preview.wav")
    sf.write(p, wav, SR)
    return {"url": "/api/audio/output/clone/preview.wav", "text": text,
            "terms": hits, "seconds": round(len(wav) / SR, 2)}


class CloneSaveIn(BaseModel):
    ref: str
    name: str = "我的声音"
    desc: str = "我自己的声音"
    text: str = ""
    engine: str = "qwen"        # qwen 小户 / omnivoice 中户 / voxcpm 大户 / indextts 戏骨


@app.post("/api/clone/save")
def api_clone_save(inp: CloneSaveIn):
    """把这段参考收进角色库，自动编下一个号。"""
    src = _clone_path(inp.ref)
    if not os.path.exists(src):
        return JSONResponse({"error": "没有这份参考音频"}, status_code=400)
    n = 1
    existing = {c["id"] for c in cards()}
    while n in existing:
        n += 1
    dst = os.path.join(CHARS, f"{n}.wav")
    _sh.copyfile(src, dst)
    card = {"id": n, "name": (inp.name or "我的声音").strip(), "desc": inp.desc,
            "ref": os.path.basename(dst), "text": inp.text,
            "engine": inp.engine if inp.engine in ("qwen", "omnivoice", "voxcpm", "indextts") else "qwen",
            "kind": "clone",
            "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(CHARS, f"{n}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"ok": True, "card": card}


# ---------- 从视频网页抓一段声音（2026-10-30 用户要的「恶搞」功能） ----------
YTDLP = os.path.expanduser("~/.dsh/tools/bin/yt-dlp")


def _ff_run(args):
    return _sp.run(args, capture_output=True, text=True, timeout=900)


FROM_TASKS = {}


class FromUrlIn(BaseModel):
    url: str = ""


def _from_url_worker(tid, url):
    """后台抓：一边抓一边报百分比，让页面能拖个进度条。

    2026-10-30 用户要求：抓之前那种「等一会儿」太虚，得让人心里有数。
    """
    t = FROM_TASKS[tid]
    try:
        t.update(msg="正在问网站要音频…", pct=0.02)
        raw = os.path.join(CLONEDIR, "from_url_raw")
        for f in glob.glob(raw + ".*"):
            try:
                os.remove(f)
            except Exception:
                pass
        proc = _sp.Popen([YTDLP, "-f", "bestaudio", "--no-playlist", "--no-warnings",
                          "--newline", "-o", raw + ".%(ext)s", url],
                         stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = (line or "").rstrip()
            if not line:
                continue
            m = _re.search(r"\[download\]\s+([\d.]+)%", line)
            if m:
                p = float(m.group(1)) / 100.0
                t.update(pct=0.05 + p * 0.62, msg="下载中 " + m.group(1) + "%")
            elif "Destination" in line:
                t.update(msg="找到音频，开始下载…")
        proc.wait()
        got = sorted(glob.glob(raw + ".*"))
        if proc.returncode != 0 or not got:
            t.update(state="error", msg="抓不下来：这链接可能不是视频页，或者要会员/登录")
            return
        t.update(pct=0.72, msg="下载好了，正在转成能用的格式…")
        dst = os.path.join(CLONEDIR, "from_url_48k.wav")
        rr = _ff_run([_FFMPEG, "-y", "-i", got[0], "-ar", "48000", "-ac", "1",
                      "-c:a", "pcm_s16le", dst])
        if rr.returncode != 0:
            t.update(state="error", msg="这段音频转不出来")
            return
        t.update(pct=0.9, msg="在找片头该从哪儿开始…")
        data, sr = sf.read(dst)
        secs = round(len(data) / sr, 2)
        det = _ff_run([_FFMPEG, "-i", dst, "-af", "silencedetect=noise=-35dB:d=0.6",
                       "-f", "null", "-"]).stderr
        ends = [float(x) for x in _re.findall(r"silence_end: ([0-9.]+)", det)]
        start = round(ends[0], 2) if ends and ends[0] < secs - 3 else 0.0
        length = round(min(18.0, max(5.0, secs - start)), 1)
        t.update(state="done", pct=1.0, msg="抓到了：%.0f 秒" % secs,
                 result={"ref": "from_url_48k.wav", "seconds": secs,
                         "url": "/api/audio/output/clone/from_url_48k.wav",
                         "start": start, "length": length})
    except Exception as e:
        t.update(state="error", msg="出错了：" + str(e))


@app.post("/api/clone/from-url")
def api_clone_from_url(inp: FromUrlIn):
    """粘一个视频网页链接，把里面的声音拉下来（异步，可以看进度）。"""
    rawtxt = (inp.url or "").strip()
    # 2026-10-30：用户常把 B 站那行分享文字整段粘进来，要自己把链接挑出来；
    # 只收 ASCII 里合法的那一串，链接后面紧跟着中文也要能断干净
    m = _re.search(r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", rawtxt)
    if not m:
        return JSONResponse({"error": "没找到链接。把那行分享文字整个粘进来也行，我会自己挑。"},
                            status_code=400)
    url = m.group(0)
    if not os.path.exists(YTDLP):
        return JSONResponse({"error": "本机没找到 yt-dlp，抓不了网页视频"}, status_code=500)
    tid = uuid.uuid4().hex[:8]
    FROM_TASKS[tid] = {"state": "running", "pct": 0.0, "msg": "准备中"}
    threading.Thread(target=_from_url_worker, args=(tid, url), daemon=True).start()
    return {"task": tid}


@app.get("/api/clone/from-url/{tid}")
def api_clone_from_url_task(tid: str):
    return FROM_TASKS.get(tid, {"state": "missing"})


# ---------- 人声 / 背景分离（2026-11-01 用户要的：视频抓来的声音去掉背景音乐） ----------
SEP_TASKS = {}
_SEP_MODEL = None
_SEP_LOCK = threading.Lock()


class SepIn(BaseModel):
    ref: str = "cut_48k.wav"


def _sep_model():
    """模型只加载一次，放着复用（加载一次约 30 秒，之后每句只要几秒）。"""
    global _SEP_MODEL
    with _SEP_LOCK:
        if _SEP_MODEL is None:
            from demucs.pretrained import get_model as _get_demucs
            os.environ.setdefault("TORCH_HOME", os.path.join(BASE, "..", "voxcpm-tentacle", ".torch"))
            _SEP_MODEL = _get_demucs("htdemucs")
            _SEP_MODEL.eval()
    return _SEP_MODEL


def _separate_wav(src, dst):
    """把一段音频拆成人声 / 背景，只写人声那一轨（48k 单声道）。"""
    import librosa
    from demucs.apply import apply_model
    model = _sep_model()
    data, sr = sf.read(src, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != model.samplerate:
        data = librosa.resample(data, orig_sr=sr, target_sr=model.samplerate)
    wav = torch.from_numpy(np.asarray(data, dtype="float32"))
    if wav.dim() == 1:
        wav = wav[None].repeat(2, 1)          # 单声道 → 立体声（模型要两轨）
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, wav[None], device="cpu", progress=False)[0]
    out = out * ref.std() + ref.mean()
    idx = model.sources.index("vocals")
    v = out[idx].mean(0).numpy()
    v = librosa.resample(v, orig_sr=model.samplerate, target_sr=48000)
    sf.write(dst, v, 48000)
    return round(len(v) / 48000.0, 2)


def _sep_worker(tid, src):
    t = SEP_TASKS[tid]
    try:
        t.update(pct=0.08, msg="正在叫人声分离模型（第一次要等半分钟）…")
        dst = os.path.join(CLONEDIR, "sep_vocals_48k.wav")
        secs = _separate_wav(src, dst)
        t.update(state="done", pct=1.0, msg="好了：背景没了，只剩人声",
                 result={"ref": "sep_vocals_48k.wav",
                         "url": "/api/audio/output/clone/sep_vocals_48k.wav",
                         "seconds": secs})
    except Exception as e:  # noqa: BLE001
        t.update(state="error", msg="分离没成功：" + str(e))


@app.post("/api/clone/separate")
def api_clone_separate(inp: SepIn):
    """把截好的一段做「人声 / 背景」分离，只留人声。"""
    src = _clone_path(inp.ref)
    if not os.path.exists(src):
        return JSONResponse({"error": "先截一段出来再分离"}, status_code=400)
    tid = uuid.uuid4().hex[:8]
    SEP_TASKS[tid] = {"state": "running", "pct": 0.0, "msg": "准备中"}
    threading.Thread(target=_sep_worker, args=(tid, src), daemon=True).start()
    return {"task": tid}


@app.get("/api/clone/separate/{tid}")
def api_clone_separate_task(tid: str):
    return SEP_TASKS.get(tid, {"state": "missing"})


class AutoSegIn(BaseModel):
    ref: str = "from_url_48k.wav"
    start: float = 0.0
    length: float = 15.0


@app.post("/api/clone/auto-seg")
def api_clone_auto_seg(inp: AutoSegIn):
    """在你框好的范围里，自动挑「最完整的一段人声」。

    规则按用户 2026-10-30 说的：**出现在最前面、而且时间最长的那个**为准
    （两段差不多长时取靠前那段），最长不超过 15 秒。

    能做到的：跳过框里的长停顿、旁白插话、前头的一段音乐。
    做不到的：两个人**同时**说话、或者人声一直压着音乐 —— 那要声源分离，本机没有模型。
    """
    src = _clone_path(inp.ref)
    if not os.path.exists(src):
        return JSONResponse({"error": "没有这份音频"}, status_code=400)
    st = max(0.0, float(inp.start))
    ln = max(1.0, min(120.0, float(inp.length)))
    tmp = os.path.join(CLONEDIR, "_autoseg16k.wav")
    r = _ff_run([_FFMPEG, "-y", "-ss", "%.2f" % st, "-t", "%.2f" % ln, "-i", src,
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp])
    if r.returncode != 0:
        return {"start": round(st, 2), "length": round(min(15.0, ln), 1), "picked": False}
    y, sr = sf.read(tmp)
    if y.ndim > 1:
        y = y.mean(1)
    hop = max(1, int(sr * 0.05))
    n = len(y) // hop
    if n < 4:
        return {"start": round(st, 2), "length": round(min(15.0, ln), 1), "picked": False}
    frames = y[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(0.008, float(np.percentile(rms, 75)) * 0.22)
    on = rms > thr
    # 先把「短短的一口气」合并掉：中间 0.45 秒以内的空隙不算断
    #（不合并的话，正常说话里的逗号停顿会把一整段切成一节一节的）
    gap = max(1, int(0.45 / 0.05))
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if i > 0 and j < n and (j - i) < gap:
                on[i:j] = True
            i = j
        else:
            i += 1
    best = None
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            dur = (j - i) * hop / sr
            # 明显更长才换 —— 两段差不多长时，保留靠前的那段
            if best is None or dur > best[1] + 0.4:
                best = (i * hop / sr, dur)
            i = j
        else:
            i += 1
    if best is None:
        return {"start": round(st, 2), "length": round(min(15.0, ln), 1), "picked": False}
    s0, d0 = best
    if d0 < 5.0:
        # 框里太零碎（最长的一段还不到 5 秒）：别硬挑，用开头 15 秒，并且告诉用户
        return {"start": round(st, 2), "length": round(min(15.0, ln), 1),
                "picked": False, "found": round(d0, 1), "messy": True}
    out_len = round(min(15.0, d0), 1)
    return {"start": round(st + s0, 2), "length": out_len, "picked": True,
            "found": round(d0, 1)}


class CutIn(BaseModel):
    ref: str = "from_url_48k.wav"
    start: float = 0.0
    length: float = 15.0


@app.post("/api/clone/cut")
def api_clone_cut(inp: CutIn):
    """截一小段：克隆参考 10–20 秒就够。"""
    src = _clone_path(inp.ref)
    if not os.path.exists(src):
        return JSONResponse({"error": "没有这份音频"}, status_code=400)
    st = max(0.0, float(inp.start))
    # 2026-10-30 用户定的安全机制：参考音最多 15 秒（前端也限了，这里再兜一层）
    ln = max(2.0, min(15.0, float(inp.length)))
    dst = os.path.join(CLONEDIR, "cut_48k.wav")
    r = _ff_run([_FFMPEG, "-y", "-ss", f"{st:.2f}", "-t", f"{ln:.2f}", "-i", src,
                 "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst])
    if r.returncode != 0:
        return JSONResponse({"error": "截不出来",
                             "detail": (r.stderr or "")[-300:]}, status_code=400)
    data, sr = sf.read(dst)
    return {"ref": "cut_48k.wav", "seconds": round(len(data) / sr, 2),
            "url": "/api/audio/output/clone/cut_48k.wav"}


# ================= 用文字描述捏新嗓子（从旧界面搬过来） =================
DESIGN_TEXTS = [
    "今天的活儿，先把手上的事情理一理，别急着往我这儿推。",
    "我这个人说话直，有什么不对的地方，你当场就可以指出来。",
    "你别站那么远，过来一点，把刚才那张单子再给我看一眼。",
    "这件事说起来也简单，就是一步一步来，中间别漏了哪一步。",
]


class DesignIn(BaseModel):
    desc: str


@app.post("/api/design")
def api_design(inp: DesignIn):
    """按一句文字描述捏 3 条候选（VoxCPM design，不需要参考音频）。"""
    desc = (inp.desc or "").strip()
    if not desc:
        return JSONResponse({"error": "先写一句描述：他是谁、多大年纪、声音什么样"}, status_code=400)
    text = random.choice(DESIGN_TEXTS)
    cands = []
    for k in range(3):
        torch.manual_seed((int(time.time() * 1000) + k * 7919) % (2 ** 31))
        with LOCK:
            wav = _vox().generate(text=f"({desc}){text}", cfg_value=2.2, inference_timesteps=16)
        name = f"候选{k}.wav"
        sf.write(os.path.join(CLONEDIR, name), wav, SR)
        cands.append({"url": "/api/audio/output/clone/" + name,
                      "seconds": round(len(wav) / SR, 2)})
    return {"candidates": cands, "text": text}


class DesignOkIn(BaseModel):
    desc: str
    text: str = ""
    name: str = ""
    cand: int = 0


@app.post("/api/design/confirm")
def api_design_confirm(inp: DesignOkIn):
    """把当前候选定成正式角色（自动编号）。"""
    src = os.path.join(CLONEDIR, f"候选{inp.cand}.wav")
    if not os.path.exists(src):
        return JSONResponse({"error": "还没有这条候选，先捏一次"}, status_code=400)
    n = 1
    existing = {c["id"] for c in cards()}
    while n in existing:
        n += 1
    dst = os.path.join(CHARS, f"{n}.wav")
    _sh.copyfile(src, dst)
    card = {"id": n, "name": (inp.name or "").strip(), "desc": inp.desc.strip(),
            "kind": "design", "engine": "voxcpm",
            "ref": os.path.basename(dst), "text": inp.text,
            "created": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(CHARS, f"{n}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"ok": True, "card": card}


# ================= 导出包（给同学进 CapCut 用） =================
import zipfile as _zip
import shutil as _sh2

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
EXPORTDIR = os.path.join(OUTDIR, "export")
os.makedirs(EXPORTDIR, exist_ok=True)


def _ff_args(fmt):
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", "192k"]
    if fmt == "wav":
        return ["-c:a", "pcm_s16le", "-ar", "48000"]
    return ["-c:a", "aac", "-b:a", "192k"]


def _dur(path):
    r = _sp.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class PackIn(BaseModel):
    name: str = "导出包"
    files: list = []          # 分句 wav 文件名，按剧本顺序
    lines: list = []          # [{no, name, text}]，用来写清单和字幕
    format: str = "mp3"
    srt: bool = True
    marks: dict = {}          # 视频打点：{"句号": 秒}
    # 2026-11-02 用户定：按角色分轨，**让用户自己选**，不替他定
    #   none    不分轨（默认，跟以前一样）
    #   compact 紧凑版：只留这个角色的话，挨着串起来
    #   aligned 对齐版：每条轨和整部一样长，话落在原来的时间点上
    #   both    两个都出
    tracks: str = "none"


def _track_label(fname, ln):
    """从分句文件名里认出「几号 · 谁」——文件名是排演时定死的，最可信。

    形如 001_1号_达老师v2.wav → "1号_达老师v2"
    认不出编号就退回剧本里的角色名。
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) >= 3 and parts[1].endswith("号"):
        return parts[1] + "_" + "_".join(parts[2:])
    if len(parts) >= 2 and parts[1].endswith("号"):
        return parts[1]
    return (ln.get("name") or "未命名")


def _ts(sec):
    m, s = divmod(float(sec), 60)
    return f"{int(m):02d}:{s:06.3f}"


@app.post("/api/export/package")
def api_export_package(inp: PackIn):
    d = os.path.join(EXPORTDIR, _safe(inp.name))
    if os.path.isdir(d):
        _sh2.rmtree(d)
    os.makedirs(d, exist_ok=True)
    fmt = inp.format if inp.format in ("m4a", "mp3", "wav") else "mp3"
    rows, subs = [], []
    t = 0.0
    idx = 0
    # 2026-11-02：按角色分轨要用的料，边转码边记下来
    # (角色标签, 起点秒, 时长秒, 句号)
    per_line = []
    for f in inp.files:
        src = os.path.join(OUTDIR, os.path.basename(f))
        if not os.path.exists(src):
            continue
        idx += 1
        ln = inp.lines[idx - 1] if idx - 1 < len(inp.lines) else {}
        # 编号用剧本里的句号（cue ID），分场排演导出时才不会又从头数
        no = str(ln.get("no", idx))
        try:
            no_int = int(no)
        except Exception:
            no_int = idx
        out = os.path.join(d, f"{no_int:03d}.{fmt}")
        r = _sp.run([FFMPEG, "-y", "-i", src, *_ff_args(fmt), out],
                    capture_output=True, text=True)
        if r.returncode != 0:
            continue
        sec = _dur(out)
        who = ln.get("name", "")
        what = ln.get("text", "")
        stamp = _ts(inp.marks[no]) if inp.marks and no in inp.marks else "--"
        rows.append(f"{no_int:03d} | {who}：{what} | {stamp}")
        subs.append((t, t + sec, f"{who}：{what}".strip("：")))
        per_line.append((_track_label(f, ln), t, sec, no_int))
        t += sec

    with open(os.path.join(d, "Manifest.txt"), "w", encoding="utf-8") as f:
        f.write("# ID | 台词 | 时间戳（-- 表示这一句还没在视频里打点）\n")
        f.write("\n".join(rows) + "\n")

    if inp.srt and subs:
        with open(os.path.join(d, "字幕.srt"), "w", encoding="utf-8") as f:
            for i, (a, b, txt) in enumerate(subs, 1):
                f.write(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{txt}\n\n")

    # ---------- 按角色分轨（用户自己选要哪种，不替他定） ----------
    track_names = []
    want = (inp.tracks or "none").strip().lower()
    if want in ("compact", "aligned", "both") and per_line:
        track_names = _write_role_tracks(d, per_line, t, want, fmt)

    if idx == 0:
        return JSONResponse({"error": "一条音频都没找到，先排演一遍再导出"}, status_code=400)
    zpath = os.path.join(EXPORTDIR, _safe(inp.name) + ".zip")
    with _zip.ZipFile(zpath, "w", _zip.ZIP_DEFLATED) as z:
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isdir(p):
                for m in sorted(os.listdir(p)):          # 按角色/ 里面的也要进 zip
                    z.write(os.path.join(p, m), os.path.join(n, m))
            else:
                z.write(p, n)
    return {"ok": True, "count": idx, "seconds": round(t, 2),
            "dir": os.path.basename(d),
            "tracks": track_names,
            "url": "/api/export/download/" + _safe(inp.name)}


def _write_role_tracks(d, per_line, total, want, fmt):
    """同一个角色的句子挑出来，做成一条轨。用户 2026-11-02 要的。

    compact 只留他的话，挨着串起来 —— 单独听、单独检查气口用；
    aligned 每条轨和整部一样长，他的话落在原来的时间点上 ——
            几条一起拖进 DAW，位置不用挪，叠起来就是完整的一出戏。
    ⚠️ 对齐版按整数采样数放位置，避免浮点误差让轨与轨错开几个采样。
    """
    from collections import OrderedDict
    groups = OrderedDict()
    for label, start, sec, no_int in per_line:
        groups.setdefault(label, []).append((start, sec, no_int))

    tdir = os.path.join(d, "按角色")
    os.makedirs(tdir, exist_ok=True)
    out_names, notes = [], ["# 按角色分轨", ""]
    sr = 48000
    total_int = int(round(total * sr))

    for label, items in groups.items():
        safe = _safe(label).replace("/", "_")
        # ---- 紧凑版 ----
        if want in ("compact", "both"):
            chunks, used = [], []
            for start, sec, no_int in items:
                src = os.path.join(d, f"{no_int:03d}.{fmt}")
                if not os.path.exists(src):
                    continue
                data, _ = sf.read(src, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                chunks.append(data); used.append(no_int)
            if chunks:
                merged = np.concatenate(chunks)
                name = f"紧凑_{safe}.{fmt}"
                _write_audio(os.path.join(tdir, name), merged, sr, fmt)
                out_names.append("按角色/" + name)
                notes.append(f"紧凑_{safe}：{len(used)} 句、{len(merged)/sr:.1f} 秒，"
                             "第 " + "、".join(f"{n:03d}" for n in used) + " 句")
        # ---- 对齐版 ----
        if want in ("aligned", "both"):
            track = np.zeros(total_int, dtype="float32")
            placed = 0
            for start, sec, no_int in items:
                src = os.path.join(d, f"{no_int:03d}.{fmt}")
                if not os.path.exists(src):
                    continue
                data, _ = sf.read(src, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                a = int(round(start * sr))
                b = min(a + len(data), total_int)
                if b > a:
                    track[a:b] += data[:b - a]
                    placed += 1
            if placed:
                name = f"对齐_{safe}.{fmt}"
                _write_audio(os.path.join(tdir, name), track, sr, fmt)
                out_names.append("按角色/" + name)
                notes.append(f"对齐_{safe}：{placed} 句、全长 {total_int/sr:.1f} 秒"
                             "（和整部一样长，话在原来的位置上）")

    with open(os.path.join(tdir, "分轨说明.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")
        f.write("\n对齐版：几条轨一起拖进 DAW，位置不用挪，叠起来就是完整的一出戏。\n")
        f.write("紧凑版：只有这个角色的话，一句接一句，适合单独听、单独检查。\n")

    # 2026-11-02 用户要的：送进 ACE Studio 时，一个角色一条轨，
    # 但轨上要**变成一块一块**（连续说的合成一块），位置保持原样。
    # 把片段算好存下来，送的时候直接照着放，不用再猜哪儿是静音。
    pieces = {"total": round(total_int / sr, 2), "roles": []}
    for label, items in groups.items():
        safe = _safe(label).replace("/", "_")
        segs = _merge_segments(items)          # [[起点秒, 时长秒], ...]
        if not segs:
            continue
        pieces["roles"].append({
            "label": label,
            "safe": safe,
            "aligned": f"对齐_{safe}.{fmt}",
            "segments": [[round(a, 3), round(b, 3)] for a, b in segs],
        })
    try:
        with open(os.path.join(tdir, "片段.json"), "w", encoding="utf-8") as f:
            json.dump(pieces, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return out_names


def _merge_segments(items, gap=0.4):
    """同一角色的句子，间隔小于 gap 秒的算「连着说」，合成一块。

    用户 2026-11-02 定：颗粒度是「连续说的合成一块」，不是一句一块。
    """
    items = sorted(items, key=lambda x: x[0])
    segs = []
    for start, sec, _no in items:
        if segs and start - (segs[-1][0] + segs[-1][1]) <= gap:
            segs[-1][1] = start + sec - segs[-1][0]      # 拉长到盖住这一句
        else:
            segs.append([start, sec])
    return segs


def _write_audio(path, data, sr, fmt):
    """写一条分轨。wav 直接落；mp3 / m4a 先落临时 wav 再转。"""
    if fmt == "wav":
        sf.write(path, data, sr)
        return
    tmp = path + ".tmp.wav"
    sf.write(tmp, data, sr)
    _sp.run([FFMPEG, "-y", "-i", tmp, *_ff_args(fmt), path],
            capture_output=True, text=True)
    try:
        os.remove(tmp)
    except Exception:
        pass


@app.get("/api/export/download/{name}")
def api_export_download(name: str):
    p = os.path.join(EXPORTDIR, _safe(name) + ".zip")
    if not os.path.exists(p):
        return JSONResponse({"error": "还没导出"}, status_code=404)
    from urllib.parse import quote
    fn = quote(_safe(name) + ".zip")
    return FileResponse(p, media_type="application/zip",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fn}"})


@app.get("/api/ppt/bundle/{name}")
def api_ppt_bundle(name: str):
    """按页序打包：音频 + 术语表配置 + 清单，给「课后发给学生」用。"""
    d = _pdir(name)
    adir = os.path.join(d, "audio")
    if not os.path.isdir(adir):
        return JSONResponse({"error": "这一册还没生成音频"}, status_code=404)
    zp = os.path.join(PPTROOT, _safe(name) + "_音频包.zip")
    n = 0
    with _zip.ZipFile(zp, "w", _zip.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(adir)):
            if f.lower().endswith((".m4a", ".mp3", ".wav")):
                z.write(os.path.join(adir, f), f)     # 文件名就是页号
                n += 1
        gpath = os.path.join(BASE, "pptunit", "glossary.json")
        if os.path.exists(gpath):
            z.write(gpath, "术语表.json")
        mpath = os.path.join(d, "manifest.json")
        if os.path.exists(mpath):
            z.write(mpath, "清单.json")
    if n == 0:
        return JSONResponse({"error": "这一册还没生成音频"}, status_code=404)
    from urllib.parse import quote
    fn = quote(_safe(name) + "_音频包.zip")
    return FileResponse(zp, media_type="application/zip",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fn}"})


# ================= 剧场 / 情绪试音（转发给 IndexTTS 常驻服务 :7866） =================
import urllib.request as _urlreq

INDEXTTS = "http://127.0.0.1:7866"
EMO_NAMES = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


class EmoSayIn(BaseModel):
    text: str
    ref: str
    lang: str = "zh"
    emotion: str = ""        # "angry:0.8,sad:0.3"
    emotion_text: str = ""   # 或者一句话描述
    emotion_ref: str = ""    # 或者一段情绪参考音（只借语气，音色仍看 ref）
    emo_alpha: float = 0.8
    speed: float = 1.0


_ITS = {"p": None}


def _theater_ready():
    try:
        with _urlreq.urlopen(INDEXTTS + "/health", timeout=3) as r:
            return bool(json.loads(r.read()).get("ready"))
    except Exception:
        return False


@app.post("/api/theater/start")
def api_theater_start():
    """按需叫醒 IndexTTS（2026-10-30 用户要的：平时不占 24G，要用时慢慢拉起来）。

    它自己闲置 20 分钟会退出（indextts_server.py 里那个看门狗）。
    """
    if _theater_ready():
        return {"state": "ready"}
    p = _ITS.get("p")
    if p is not None and p.poll() is None:
        return {"state": "starting"}
    script = os.path.join(BASE, "indextts_server.py")
    if not os.path.exists(script):
        return JSONResponse({"error": "找不到 indextts_server.py"}, status_code=500)
    log = open("/tmp/indextts.log", "ab")
    try:
        _ITS["p"] = _sp.Popen([_sys.executable, script], cwd=BASE,
                              stdout=log, stderr=_sp.STDOUT, env={**os.environ})
    except Exception as e:
        return JSONResponse({"error": f"起不来：{e}"}, status_code=500)
    return {"state": "starting"}


@app.post("/api/theater/stop")
def api_theater_stop():
    """想腾地方就点它（也可以不管，20 分钟自己退）。"""
    p = _ITS.get("p")
    if p is not None and p.poll() is None:
        try:
            p.terminate()
        except Exception:
            pass
    return {"ok": True}


@app.get("/api/theater/health")
def api_theater_health():
    try:
        with _urlreq.urlopen(INDEXTTS + "/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "ready": False, "msg": f"IndexTTS 服务没起来：{e}"}


@app.post("/api/theater/say")
def api_theater_say(inp: EmoSayIn):
    ref = inp.ref
    if not os.path.isabs(ref):
        ref = os.path.join(BASE, ref)
    emo_ref = inp.emotion_ref
    if emo_ref and not os.path.isabs(emo_ref):
        emo_ref = os.path.join(BASE, emo_ref)
    body = json.dumps({"text": inp.text, "ref": ref, "lang": inp.lang,
                       "emotion": inp.emotion, "emotion_text": inp.emotion_text,
                       "emotion_ref": emo_ref,
                       "emo_alpha": inp.emo_alpha, "speed": inp.speed}).encode("utf-8")
    req = _urlreq.Request(INDEXTTS + "/say", data=body,
                          headers={"Content-Type": "application/json"})
    try:
        with _urlreq.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
    except Exception as e:
        return JSONResponse({"error": f"IndexTTS 服务出错：{e}"}, status_code=503)
    d["url"] = INDEXTTS + "/audio/" + d.get("file", "")
    return d


# ---------- 「满意了就留下来」：把这一条音频存进成品夹 ----------
KEEPDIR = os.path.join(OUTDIR, "theater", "keep")
os.makedirs(KEEPDIR, exist_ok=True)


def _trim_head(wav, sr, thresh=0.012, keep_ms=5):
    """★ 用户 2026-10-30 定：每段成品的开头必须正对 0 秒，前面不留白。
    找第一个真正有声音的采样，往前只留 5 毫秒防切掉起音，其余开头静音全部切掉。"""
    a = np.abs(wav)
    idx = np.where(a > thresh)[0]
    if not len(idx):
        return wav
    start = max(0, int(idx[0]) - int(sr * keep_ms / 1000))
    return wav[start:]


class KeepIn(BaseModel):
    file: str
    text: str = ""
    emotion: str = ""


@app.post("/api/theater/keep")
def api_theater_keep(inp: KeepIn):
    src = os.path.join(OUTDIR, "theater", os.path.basename(inp.file))
    if not os.path.exists(src):
        return JSONResponse({"error": "找不到这条音频，重新生成一次"}, status_code=404)
    head = "".join(ch for ch in (inp.text or "未命名")[:10] if ch not in '/\\:*?"<>|').strip() or "未命名"
    name = time.strftime("%m%d_%H%M%S") + "_" + head + ".wav"
    dst = os.path.join(KEEPDIR, name)
    data, sr = sf.read(src)
    sf.write(dst, _trim_head(np.asarray(data, dtype=np.float32), sr), sr)
    with open(os.path.join(KEEPDIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"text": inp.text, "emotion": inp.emotion,
                   "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    return {"ok": True, "name": name, "count": len(glob.glob(os.path.join(KEEPDIR, "*.wav")))}


@app.get("/api/theater/keep")
def api_theater_keep_list():
    files = sorted(glob.glob(os.path.join(KEEPDIR, "*.wav")), key=os.path.getmtime, reverse=True)
    return {"items": [{"name": os.path.basename(f),
                       "url": "/api/theater/keep/file/" + os.path.basename(f)} for f in files[:50]]}


@app.get("/api/theater/keep/file/{name}")
def api_theater_keep_file(name: str):
    p = os.path.join(KEEPDIR, os.path.basename(name))
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个文件"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


# ================= 送到 DAW：Live / ACE Studio =================
# 用户 2026-10-30：「给 live 和 ACE 搭一个桥」，只送音频，不碰速度（两边都自己测速/按原速放）。
PUSH_LIVE = os.path.normpath(os.path.join(BASE, "..", "ableton-mysongwriter", "push_to_live.py"))
ACE_CLI = "/Applications/ACE Studio.app/Contents/Helpers/acestudio-cli"


class SendIn(BaseModel):
    file: str
    track: str = "VoiceCaster"
    at_current: bool = True     # 以 DAW 里的播放指针为准（2026-10-30 用户要的）


def _send_path(inp):
    p = inp.file
    if not os.path.isabs(p):
        p = os.path.join(KEEPDIR, os.path.basename(p))
    return p


@app.post("/api/send/live")
def api_send_live(inp: SendIn):
    p = _send_path(inp)
    if not os.path.exists(p):
        return JSONResponse({"error": "找不到这个音频"}, status_code=404)
    if not os.path.exists(PUSH_LIVE):
        return JSONResponse({"error": "找不到 Live 桥脚本：" + PUSH_LIVE}, status_code=500)
    args = [_sys.executable, PUSH_LIVE, "--audio", p, "--track", inp.track, "--json"]
    if inp.at_current:
        args.append("--at-current")
    try:
        r = _sp.run(args, capture_output=True, text=True, timeout=180)
    except _sp.TimeoutExpired:
        return JSONResponse({"error": "Live 没回话：它开了吗？偏好设置里把 AbletonMySongwriter 选成控制面了吗？"},
                            status_code=504)
    if r.returncode != 0:
        return JSONResponse({"error": "Live 没收下：" + ((r.stderr or r.stdout or "")[-400:])},
                            status_code=500)
    return {"ok": True, "out": (r.stdout or "").strip()[-400:]}


@app.post("/api/send/ace")
def api_send_ace(inp: SendIn):
    p = _send_path(inp)
    if not os.path.exists(p):
        return JSONResponse({"error": "找不到这个音频"}, status_code=404)
    if not os.path.exists(ACE_CLI):
        return JSONResponse({"error": "找不到 ACE Studio（没装？）"}, status_code=500)
    env = dict(os.environ, ACESTUDIO_CLI_NO_AGENT_HINT="1")
    # 先读 ACE 的光标（marker line）位置，再按它放
    pos = 0.0
    if inp.at_current:
        try:
            r0 = _sp.run([ACE_CLI, "caret", "get", "--json"],
                         capture_output=True, text=True, env=env, timeout=30)
            pos = float(json.loads(r0.stdout).get("sec") or 0)
        except Exception:
            pos = 0.0
    args = [ACE_CLI, "import", "file", "--path", p]
    if pos > 0.001:
        args += ["--pos", "%.3fs" % pos]
    try:
        r = _sp.run(args, capture_output=True, text=True, env=env, timeout=180)
    except _sp.TimeoutExpired:
        return JSONResponse({"error": "ACE 没回话：它开着吗？"}, status_code=504)
    if r.returncode != 0:
        return JSONResponse({"error": "ACE 没收下：" + ((r.stderr or r.stdout or "")[-400:])},
                            status_code=500)
    return {"ok": True, "out": (r.stdout or "").strip()[-400:]}


# ================= 送到 Musio Studio =================
# 2026-11-03：整段放在 Studio 的光标处；有句表（整段.wav.sentences.json）就按句切开。
MUSIO_BRIDGE = os.path.normpath(os.path.join(BASE, "..", "musio-bridge", "musio_bridge.py"))


@app.post("/api/send/musio")
def api_send_musio(inp: SendIn):
    p = _send_path(inp)
    if not os.path.exists(p):
        return JSONResponse({"error": "找不到这个音频"}, status_code=404)
    if not os.path.exists(MUSIO_BRIDGE):
        return JSONResponse({"error": "找不到 Musio 桥：" + MUSIO_BRIDGE}, status_code=500)
    splits = []
    meta = p + ".sentences.json"
    if os.path.exists(meta):
        try:
            starts = json.load(open(meta, encoding="utf-8")).get("starts") or []
            splits = starts[1:]          # 第一句从 0 秒起，不用切
        except Exception:
            splits = []
    args = [_sys.executable, MUSIO_BRIDGE, "send-audio", "--path", p,
            "--track", inp.track or "VoiceCaster"]
    if splits:
        args += ["--split", ",".join(str(x) for x in splits)]
    try:
        r = _sp.run(args, capture_output=True, text=True, timeout=180)
    except _sp.TimeoutExpired:
        return JSONResponse({"error": "Musio 没回话：Studio 开着吗？设置 → Agents 打开了吗？"},
                            status_code=504)
    if r.returncode != 0:
        return JSONResponse({"error": "Musio 没收下：" + ((r.stderr or r.stdout or "")[-400:])},
                            status_code=500)
    return {"ok": True, "out": (r.stdout or "").strip()[-400:], "cuts": len(splits)}


class AceTracksIn(BaseModel):
    dir: str = ""          # 导出包目录名，output/export/<dir> 里有「按角色/片段.json」


@app.post("/api/send/ace-tracks")
def api_send_ace_tracks(inp: AceTracksIn):
    """把「按角色分轨」的结果送进 ACE Studio，每个角色一条轨，轨上是一块一块的。

    用户 2026-11-02 的想法：既然能按角色对齐时间线，就让 ACE 自动切成块，
    空着的地方不占位，每块还能单独拖。
    - 一个角色 = 一条 audio 轨；
    - 连续说的话 = 一块（片段清单导出时就算好了，见 _merge_segments）；
    - 每块仍停在原来的时间点上（--pos = --clip-in = 原起点）。
    """
    d = os.path.join(EXPORTDIR, _safe(inp.dir))
    pj = os.path.join(d, "按角色", "片段.json")
    if not os.path.exists(pj):
        return JSONResponse({"error": "这个导出包里没有分轨信息——导出时把「分轨」选成紧凑/对齐/两个都要"}, status_code=400)
    if not os.path.exists(ACE_CLI):
        return JSONResponse({"error": "找不到 ACE Studio（没装？）"}, status_code=500)
    with open(pj, encoding="utf-8") as f:
        info = json.load(f)
    env = dict(os.environ, ACESTUDIO_CLI_NO_AGENT_HINT="1")

    done, errors = [], []
    for role in info.get("roles", []):
        src = os.path.join(d, "按角色", role.get("aligned", ""))
        if not os.path.exists(src):
            errors.append(f"{role.get('label')}：包里没有 {role.get('aligned')}")
            continue
        try:
            r = _sp.run([ACE_CLI, "track", "create", "--type", "audio",
                         "--name", role.get("label", "角色"), "--json"],
                        capture_output=True, text=True, env=env, timeout=90)
            tu = json.loads(r.stdout)["trackUuid"]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{role.get('label')}：建轨失败 {e}")
            continue
        clips = 0
        for a, b in role.get("segments", []):
            args = [ACE_CLI, "import", "file", "--path", src,
                    "--track-uuid", tu,
                    "--pos", "%.3fs" % float(a),
                    "--clip-in", "%.3fs" % float(a),
                    "--dur", "%.3fs" % float(b),
                    "--on-occupied", "fail", "--json"]
            try:
                rr = _sp.run(args, capture_output=True, text=True, env=env, timeout=180)
            except _sp.TimeoutExpired:
                errors.append(f"{role.get('label')} 第 {clips+1} 块：ACE 没回话")
                break
            if rr.returncode == 0:
                clips += 1
            else:
                errors.append(f"{role.get('label')} 第 {clips+1} 块："
                              + ((rr.stderr or rr.stdout or "")[-120:]))
        done.append({"label": role.get("label"), "track": tu, "clips": clips})

    return {"ok": True, "roles": done, "errors": errors,
            "summary": "、".join(f"{x['label']} {x['clips']} 块" for x in done)}


class LiveTracksIn(BaseModel):
    dir: str = ""


@app.post("/api/send/live-tracks")
def api_send_live_tracks(inp: LiveTracksIn):
    """按角色送进 Ableton Live：一角色一轨，轨上是一块一块，位置回到原秒数。

    2026-11-02 用户要的（跟送 ACE 同一套想法）。
    Live 侧靠 Remote Script 的 pos_sec 支持：给了它，块就不再挨着排，
    而是回到片段清单里记的那个秒数上（内部按当前工程速度换算成拍）。
    """
    d = os.path.join(EXPORTDIR, _safe(inp.dir))
    pj = os.path.join(d, "按角色", "片段.json")
    if not os.path.exists(pj):
        return JSONResponse({"error": "这个导出包里没有分轨信息——导出时把「分轨」选上"}, status_code=400)
    if not os.path.exists(PUSH_LIVE):
        return JSONResponse({"error": "找不到 Live 桥脚本：" + PUSH_LIVE}, status_code=500)
    with open(pj, encoding="utf-8") as f:
        info = json.load(f)

    done, errors = [], []
    for role in info.get("roles", []):
        src = os.path.join(d, "按角色", role.get("aligned", ""))
        if not os.path.exists(src):
            errors.append(f"{role.get('label')}：包里没有 {role.get('aligned')}")
            continue
        # 2026-11-02 实测结论：Live 那边**搬片段时会丢掉裁剪**
        # （Session 里设好 7 秒，搬到编曲视图仍按源全长 98.6 秒算，
        #  多块互相压短就成了首尾相接的长条）。
        # ⇒ 这里先把每块切成独立小文件，文件本身多长，放上去就多长，不靠裁剪。
        safe0 = _safe(role.get("label", "角色")).replace("/", "_")
        segdir = os.path.join(d, "按角色", "片段文件")
        os.makedirs(segdir, exist_ok=True)
        secs = []
        for i, ab in enumerate(role.get("segments", [])):
            a, b = float(ab[0]), float(ab[1])
            fn = os.path.join(segdir, "%s_块%02d.wav" % (safe0, i + 1))
            rr = _sp.run([FFMPEG, "-y", "-ss", "%.3f" % a, "-t", "%.3f" % b,
                          "-i", src, "-ar", "48000", "-ac", "1",
                          "-c:a", "pcm_s16le", fn],
                         capture_output=True, text=True, timeout=120)
            if rr.returncode != 0 or not os.path.exists(fn):
                errors.append("%s 第 %d 块：切不出来" % (role.get("label"), i + 1))
                continue
            secs.append({"name": "%s-%d" % (role.get("label", "角色"), i + 1),
                         "path": fn, "pos_sec": a})
        if not secs:
            errors.append(f"{role.get('label')}：一块都没切出来")
            continue
        tmp = os.path.join(d, "按角色", "_live_sections.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(secs, f, ensure_ascii=False)
        try:
            r = _sp.run([_sys.executable, PUSH_LIVE, "--audio", secs[0]["path"],
                         "--track", role.get("label", "角色"),
                         "--sections-json", tmp, "--no-tempo", "--json"],
                        capture_output=True, text=True, timeout=600)
        except _sp.TimeoutExpired:
            errors.append(f"{role.get('label')}：Live 没回话（它开着吗？控制面选了吗？）")
            continue
        if r.returncode == 0:
            try:
                res = json.loads(r.stdout)
                n = len((res.get("result") or {}).get("clips") or [])
            except Exception:  # noqa: BLE001
                n = len(secs)
            done.append({"label": role.get("label"), "clips": n})
        else:
            errors.append(f"{role.get('label')}：" + ((r.stderr or r.stdout or "")[-150:]))

    return {"ok": True, "roles": done, "errors": errors,
            "summary": "、".join(f"{x['label']} {x['clips']} 块" for x in done)}



# ================= 批量导入现成录音（学生发来的手机音频） =================
class BatchIn(BaseModel):
    items: list        # [{name: "张三.m4a", data: base64}]


@app.post("/api/characters/import-batch")
def api_import_batch(inp: BatchIn):
    """一次收一堆录音：文件名当角色名，自动转 48k 单声道、自动编号入库。"""
    added, failed = [], []
    n = max([c["id"] for c in cards()] or [0])
    for it in inp.items:
        try:
            raw = base64.b64decode(it.get("data", ""))
            ext = os.path.splitext(it.get("name", ""))[1].lower() or ".wav"
            tmp = os.path.join(CLONEDIR, "batch_in" + ext)
            with open(tmp, "wb") as f:
                f.write(raw)
            n += 1
            dst = os.path.join(CHARS, f"{n}.wav")
            r = _sp.run([_FFMPEG, "-y", "-i", tmp, "-ar", "48000", "-ac", "1",
                         "-c:a", "pcm_s16le", dst], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(dst):
                n -= 1
                failed.append(it.get("name", "?"))
                continue
            name = os.path.splitext(os.path.basename(it.get("name", "")))[0].strip() or f"{n}号"
            card = {"id": n, "name": name, "desc": "导入的录音", "ref": f"{n}.wav",
                    "text": "", "created": time.strftime("%Y-%m-%d %H:%M")}
            with open(os.path.join(CHARS, f"{n}.json"), "w", encoding="utf-8") as f:
                json.dump(card, f, ensure_ascii=False, indent=2)
            added.append(f"{n} 号 {name}")
        except Exception:
            failed.append(it.get("name", "?"))
    return {"ok": True, "added": added, "failed": failed}


# ---------- Cubase 收件箱 ----------
# Cubase 没有 Live(Remote Script)/ACE(CLI) 那种外部接口，改不了「点一下落指针」。
# 用户 2026-10-30 要「给 cubase 做一下吧」——能做的是：把音频放进固定收件箱，
# 他在 Cubase 的 MediaBay 里收藏这个文件夹，拖一下即用。
CUBASE_INBOX = os.path.join(OUTDIR, "cubase-inbox")
os.makedirs(CUBASE_INBOX, exist_ok=True)


@app.post("/api/send/cubase")
def api_send_cubase(inp: SendIn):
    p = _send_path(inp)
    if not os.path.exists(p):
        return JSONResponse({"error": "找不到这个音频"}, status_code=404)
    dst = os.path.join(CUBASE_INBOX, os.path.basename(p))
    _sh.copyfile(p, dst)
    return {"ok": True, "path": dst, "folder": CUBASE_INBOX}


@app.get("/api/cubase/inbox")
def api_cubase_inbox():
    files = sorted(glob.glob(os.path.join(CUBASE_INBOX, "*.wav")) +
                   glob.glob(os.path.join(CUBASE_INBOX, "*.mp3")),
                   key=os.path.getmtime, reverse=True)
    return {"folder": CUBASE_INBOX, "count": len(files),
            "items": [os.path.basename(f) for f in files[:50]]}


# ================= .cast 角色包：一部广播剧的全套嗓子 =================
# 用户 2026-10-30：专门保存「为这部剧捏好的一堆角色」，载入就全部回来。
class CastIn(BaseModel):
    data: str


@app.get("/api/cast/export")
def api_cast_export():
    cs = cards()
    if not cs:
        return JSONResponse({"error": "还没有角色，先录一个"}, status_code=400)
    zp = os.path.join(OUTDIR, "角色包.cast")
    with _zip.ZipFile(zp, "w", _zip.ZIP_DEFLATED) as z:
        meta = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chars": [{"id": c["id"], "name": c.get("name", ""), "desc": c.get("desc", ""),
                           "ref": c.get("ref", ""), "text": c.get("text", "")} for c in cs]}
        z.writestr("cast.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for c in cs:
            f = os.path.join(CHARS, c.get("ref", ""))
            if os.path.exists(f):
                z.write(f, "audio/" + os.path.basename(f))
    from urllib.parse import quote
    fn = quote("角色包_" + time.strftime("%m%d") + ".cast")
    return FileResponse(zp, media_type="application/octet-stream",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{fn}"})


@app.post("/api/cast/import")
def api_cast_import(inp: CastIn):
    raw = base64.b64decode(inp.data)
    try:
        zf = _zip.ZipFile(io.BytesIO(raw))
        meta = json.loads(zf.read("cast.json"))
    except Exception as e:
        return JSONResponse({"error": "不是有效的 .cast 文件：" + str(e)[:120]}, status_code=400)
    cs = meta.get("chars", [])
    if not cs:
        return JSONResponse({"error": "这个 .cast 里没有角色"}, status_code=400)
    existing = {c["id"] for c in cards()}
    added, renumbered = 0, []
    for c in cs:
        cid = int(c.get("id") or 0)
        if not cid:
            continue
        if cid in existing:
            new = (max(existing) if existing else 0) + 1
            renumbered.append(f"{cid}→{new}")
            cid = new
        src = "audio/" + os.path.basename(c.get("ref", ""))
        if src not in zf.namelist():
            continue
        dst = os.path.join(CHARS, f"{cid}.wav")
        with zf.open(src) as f, open(dst, "wb") as o:
            _sh.copyfileobj(f, o)
        card = {"id": cid, "name": c.get("name", ""), "desc": c.get("desc", ""),
                "ref": f"{cid}.wav", "text": c.get("text", ""),
                "created": time.strftime("%Y-%m-%d %H:%M")}
        with open(os.path.join(CHARS, f"{cid}.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        existing.add(cid)
        added += 1
    return {"ok": True, "added": added, "renumbered": renumbered}


@app.get("/favicon.ico")
def _favicon():
    # 没放图标文件：回一个空的，省得浏览器每次开页面都在日志里留两条 404
    from fastapi import Response
    return Response(status_code=204)


@app.middleware("http")
async def _no_store_page(request, call_next):
    """页面本身永远不吃缓存。

    2026-10-30 踩过：改完前端，用户浏览器还给着旧页面，报的 bug 在新代码里早就不存在。
    """
    resp = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp


# ================= 朗读单元（读故事 / 有声书）=================
# 2026-11-01：接替原来的「情绪试音」——那是给大户 IndexTTS 的试验页，大户没装、平时空转。
# 结构按用户口述的规格：
#   编辑框（自由写）→ 投递 → 朗读框（只读）；选中右键改角色/写一句指令，光标处右键插停顿、笑声。
# 底层：本机千问 3 TTS（:3910，官方音色 / 克隆 / instruct）；
#       方言段按用户定的交给 Vox（VoxCPM2，靠方言书面字和提示词定调）。
READROOT = os.path.join(OUTDIR, "read")
os.makedirs(READROOT, exist_ok=True)
READTASKS = {}
NARR_TONE = "沉稳叙述感，语速中等"
READ_GAP = 0.3
READ_MAX_CHARS = 150


class ReadSpan(BaseModel):
    start: int = 0
    end: int = 0
    cid: int | str | None = None
    instruct: str = ""


class ReadPause(BaseModel):
    at: int = 0
    sec: float = 1.0


class ReadBlock(BaseModel):
    text: str = ""
    cid: int | str | None = None
    instruct: str = ""
    dialect: str = ""
    spans: list[ReadSpan] = []
    pauses: list[ReadPause] = []


class ReadIn(BaseModel):
    title: str = "朗读"
    voice: str = ""
    narr_tone: str = ""
    blocks: list[ReadBlock] = []


def _read_card(cid):
    if not cid:
        return None
    # 2026-11-01：官方九条也能直接当“人”选 —— 前端传 "o:<voice>"，
    # 这里当场造一张虚拟卡，走下面的 official 分支（千问官方音色）。
    s = str(cid).strip()
    if s.startswith("o:"):
        vid = s[2:]
        v = next((x for x in OFFICIAL_VOICES if x["id"] == vid), None)
        if not v:
            return None
        return {"id": s, "name": v["name"], "desc": v["desc"], "kind": "official",
                "voice": vid, "engine": "qwen", "ref": "", "text": ""}
    p = os.path.join(CHARS, f"{s}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _read_ref(card):
    if card and card.get("ref"):
        p = os.path.join(CHARS, card["ref"])
        if os.path.exists(p):
            return p
    for p in sorted(glob.glob(os.path.join(CHARS, "*.wav"))):
        if os.path.basename(p).startswith("_"):
            continue
        return p
    return ""


def _read_split(text):
    """长段按标点再切，免得一次生成太久（千问短句更稳）。"""
    if len(text) <= READ_MAX_CHARS:
        return [text]
    parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > READ_MAX_CHARS:
            out.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        out.append(buf)
    return out or [text]


def _read_segments(block):
    """一段拆成「念什么」和「停多久」，顺序就是朗读顺序。"""
    text = block.text or ""
    if not text.strip():
        return []
    n = len(text)
    cuts = {0, n}
    for p in block.pauses:
        cuts.add(max(0, min(n, int(p.at))))
    pausemap = {}
    for p in block.pauses:
        pausemap.setdefault(max(0, min(n, int(p.at))), []).append(float(p.sec))
    segs = []
    cs = sorted(cuts)
    for a, b in zip(cs, cs[1:]):
        piece = text[a:b]
        if piece.strip():
            bounds = {a, b}
            for s in block.spans:
                sa = max(a, min(b, int(s.start)))
                sb = max(a, min(b, int(s.end)))
                if sb > sa:
                    bounds.add(sa)
                    bounds.add(sb)
            bs = sorted(bounds)
            for x, y in zip(bs, bs[1:]):
                sub = text[x:y]
                if not sub.strip():
                    continue
                act = [s for s in block.spans
                       if min(int(s.end), y) > max(int(s.start), x)]
                cid = block.cid
                ins = []
                if (block.instruct or "").strip():
                    ins.append(block.instruct.strip())
                for s in act:
                    if s.cid is not None:
                        cid = s.cid
                    if (s.instruct or "").strip():
                        ins.append(s.instruct.strip())
                for chunk in _read_split(sub):
                    segs.append({"text": chunk, "cid": cid, "dialect": block.dialect,
                                 "instruct": "，".join(ins)})
        for sec in pausemap.get(b, []):
            segs.append({"pause": max(0.0, min(30.0, float(sec)))})
    return segs


def _read_render(seg, narr_voice, narr_tone=""):
    txt = (seg.get("text") or "").strip()
    if not txt:
        return None
    card = _read_card(seg.get("cid"))
    instruct = (seg.get("instruct") or "").strip()
    # 2026-11-01：段上没写方言就用这条声音自己的方言（声库里可以给每条声音配方言）
    dialect = (seg.get("dialect") or "").strip() or ((card.get("dialect") or "").strip() if card else "")
    # 角色卡上的默认语气（2026-11-01 加：角色槽 = 音色 + 默认指令）
    base = ((card.get("instruct") if card else "") or "").strip()
    tone = instruct or base or (narr_tone or "").strip() or NARR_TONE
    # 方言：按用户 2026-11-01 定的，交给 Vox 念
    if dialect:
        ref = _read_ref(card)
        if ref:
            ctrl = tone + f"，用{dialect}说"
            with LOCK:
                return _vox().generate(text=f"({ctrl}){txt}", reference_wav_path=ref,
                                      cfg_value=2.2, inference_timesteps=16)
    if card:
        # ★ 2026-11-04 用户定：「一般人类情况，小米 solo」→ 朗读默认中户（小米）；
        #   全局下拉强制时听全局；方言仍然在前面就交给大户（Vox）了
        _g = _global_engine()
        # ★ 小米只接外文稿；中文朗读走千问，它平时闲着
        eng = _g if _g in _GLOBAL_ENGINES else ("omnivoice" if _looks_foreign(txt) else "qwen")
        if eng == "omnivoice":
            ref = _read_ref(card)
            if ref and _omni_ready():
                try:
                    return _omni_wav(txt, ref, card.get("ref_text") or "",
                                     instruct=instruct or base or None)
                except Exception:
                    pass
            eng = "qwen"
        if eng == "qwen":
            if (card.get("kind") or "clone") == "official" and card.get("voice"):
                return _qwen_wav(txt, voice=card["voice"],
                                 instruct=instruct or base or None)
            ref = _read_ref(card)
            if not card.get("ref_text"):
                rt = _transcribe_ref(ref)
                if rt:
                    card["ref_text"] = rt
                    try:
                        _save_card(card)
                    except Exception:
                        pass
            return _qwen_wav(txt, ref=ref, ref_text=card.get("ref_text") or "",
                             instruct=instruct or base or None)
        ref = _read_ref(card)
        ctrl = instruct or base or (card.get("desc") or "") or NARR_TONE
        with LOCK:
            return _vox().generate(text=f"({ctrl}){txt}", reference_wav_path=ref,
                                  cfg_value=2.2, inference_timesteps=16)
    # 2026-11-01：旁白下拉也能挑官方音色（值是 "o:vivian"），剥掉前缀再交给千问
    nv = (narr_voice or "").strip()
    if nv.startswith("o:"):
        nv = nv[2:]
    return _qwen_wav(txt, voice=nv or "vivian",
                     instruct=instruct or tone)


def _read_run(tid, blocks, voice, narr_tone=""):
    t = READTASKS[tid]
    dirp = os.path.join(READROOT, tid)
    try:
        plan = [(bi, _read_segments(b)) for bi, b in enumerate(blocks)]
        t["total"] = sum(1 for _, segs in plan for s in segs if "text" in s)
        parts = []
        idx = 0
        for bi, segs in plan:
            if not segs:
                continue
            if parts:
                parts.append(np.zeros(int(SR * READ_GAP), dtype="float32"))
            for seg in segs:
                if "pause" in seg:
                    parts.append(np.zeros(int(SR * seg["pause"]), dtype="float32"))
                    continue
                wav = _read_render(seg, voice, narr_tone)
                if wav is None:
                    continue
                wav = np.asarray(wav, dtype="float32").reshape(-1)
                parts.append(wav)
                t["done"] += 1
                idx += 1
                try:
                    sf.write(os.path.join(dirp, f"seg_{idx:03d}.wav"), wav, SR)
                except Exception:
                    pass
        if not parts:
            raise RuntimeError("没有可念的字")
        full = np.concatenate(parts)
        sf.write(os.path.join(dirp, "full.wav"), full, SR)
        t["url"] = f"/api/audio/output/read/{tid}/full.wav"
        t["secs"] = round(len(full) / float(SR), 1)
        t["state"] = "done"
    except Exception as e:  # noqa: BLE001
        t["state"] = "error"
        t["error"] = str(e)


@app.post("/api/read/generate")
def api_read_generate(inp: ReadIn):
    blocks = [b for b in inp.blocks if (b.text or "").strip()]
    if not blocks:
        return JSONResponse({"error": "朗读框还是空的"}, status_code=400)
    if not _qwen_ready():
        return JSONResponse({"error": "千问朗读服务没开（127.0.0.1:3910）"}, status_code=503)
    tid = uuid.uuid4().hex[:8]
    os.makedirs(os.path.join(READROOT, tid), exist_ok=True)
    READTASKS[tid] = {"state": "running", "total": 0, "done": 0, "url": "", "error": ""}
    threading.Thread(target=_read_run, args=(tid, blocks, inp.voice, inp.narr_tone), daemon=True).start()
    return {"tid": tid}


@app.get("/api/read/task/{tid}")
def api_read_task(tid: str):
    return READTASKS.get(tid, {"state": "missing"})


class DetectIn(BaseModel):
    texts: list[str] = []
    min_hits: int = 2


@app.post("/api/dialect/detect")
def api_dialect_detect(inp: DetectIn):
    """朗读投递时用：每段认一下是不是方言，认出来就交给 Vox 念。"""
    return {"items": [_dial.detect(t, inp.min_hits) for t in inp.texts]}


@app.get("/api/engines/state")
def api_engines_state():
    """给下拉旁边那行小字用：现在谁在内存里、还剩多少。"""
    with _RES_LOCK:
        res = {k: round(time.time() - v) for k, v in RESIDENT.items()}
    return {"resident": res, "cap": MEM_CAP, "vox_loaded": _VOX is not None,
            "avail_gb": round(_mem_avail_gb(), 1), "floor_gb": MEM_FLOOR_GB,
            "vox_idle_sec": VOX_IDLE_SEC}


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7864, log_level="warning")
