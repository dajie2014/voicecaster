"""IndexTTS 2.5 常驻服务（端口 7866）—— 2026-10-30

为什么要有它：CLI 每次启动都要冷启动一次（实测首句 105 秒），之后每句只要 6–7 秒。
所以模型必须一直活着：启动时预热一次，之后整本台本连着跑。

跑：venv/bin/python indextts_server.py
接口：
  GET  /health
  POST /say   {text, ref, lang, emotion, emo_alpha, speed, steps}
"""
import os, sys, time, json, uuid, threading
import numpy as np

R = "/Volumes/REBIRTH/IndexTTS2.5-mlx"
MODEL_DIR = R + "/mlx-indextts2-inference/models/mlx-IndexTTS-2.5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "theater")

os.environ.setdefault("HF_HOME", R + "/.hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numcache")
# ★ 关键：MLX 0.31.1 上 CFM 的 mx.compile 会崩（cell/asyncio 报错 + 段错误），必须关
os.environ.setdefault("MLX_INDEXTTS_COMPILE_CFM", "0")
os.makedirs("/tmp/numcache", exist_ok=True)
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, R + "/mlx-indextts2-inference")

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import soundfile as sf

app = FastAPI()
ENGINE = None
LOCK = threading.Lock()
STATE = {"ready": False, "warm_s": 0.0, "msg": "starting"}
WARM_REF = None


def _boot():
    global ENGINE, WARM_REF
    t0 = time.time()
    STATE["msg"] = "loading model"
    from mlx_indextts.generate_v25 import IndexTTSv25
    ENGINE = IndexTTSv25(MODEL_DIR, qwen_model_path=MODEL_DIR.replace("mlx-IndexTTS-2.5", "qwen0.6bemo4-merge-mlx-8bit"))
    STATE["msg"] = "warming up (first call compiles MLX graphs, ~105s)"
    # 用一段短话预热：首句慢是图编译，跟内容无关
    WARM_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "characters", "_warm.wav")
    if not os.path.exists(WARM_REF):
        # 没有参考音就借一段极短的静音/点击声：只为触发图编译
        sr = 24000
        w = (np.random.randn(sr).astype(np.float32) * 0.01)
        sf.write(WARM_REF, w, sr)
    try:
        ENGINE.generate("嗯。", WARM_REF)
    except Exception as e:
        STATE["msg"] = f"warm failed: {e}"
    STATE["warm_s"] = round(time.time() - t0, 1)
    STATE["ready"] = True
    STATE["msg"] = "ready"
    print(f"[indextts] ready in {STATE['warm_s']}s", flush=True)


threading.Thread(target=_boot, daemon=True).start()

# ★ 2026-10-30：闲置自动退出。这个模型一坐就是 24G，用户不常用；
#   主服务会在需要时再把它拉起来（/api/theater/start）。
LAST_HIT = [time.time()]
IDLE_EXIT_MIN = 20


def _idle_watch():
    while True:
        time.sleep(30)
        try:
            if LOCK.locked():
                LAST_HIT[0] = time.time()      # 正在算，不算闲
            if time.time() - LAST_HIT[0] > IDLE_EXIT_MIN * 60:
                print(f"[indextts] 闲置 {IDLE_EXIT_MIN} 分钟，退出释放内存", flush=True)
                os._exit(0)
        except Exception:
            pass


threading.Thread(target=_idle_watch, daemon=True).start()


class SayIn(BaseModel):
    text: str
    ref: str
    lang: str = "zh"
    emotion: str = ""
    emotion_text: str = ""
    emotion_ref: str = ""     # 情绪参考音：只借语气，音色仍看 ref
    emo_alpha: float = 0.8
    speed: float = 1.0
    steps: int = 25


def _touch():
    LAST_HIT[0] = time.time()


@app.get("/health")
def health():
    return {"ok": True, "ready": STATE["ready"], "warm_s": STATE["warm_s"],
            "msg": STATE["msg"]}


@app.post("/say")
def say(inp: SayIn):
    _touch()
    if not STATE["ready"]:
        return JSONResponse({"error": "还在预热：" + STATE["msg"]}, status_code=503)
    if not os.path.exists(inp.ref):
        return JSONResponse({"error": "参考音不存在：" + inp.ref}, status_code=400)
    if inp.emotion_ref and not os.path.exists(inp.emotion_ref):
        return JSONResponse({"error": "情绪参考音不存在：" + inp.emotion_ref}, status_code=400)
    kw = {"language": inp.lang, "diffusion_steps": max(4, min(40, inp.steps)),
          "speed": max(0.5, min(2.0, inp.speed))}
    # 官方规定：情绪源只能有一个（参考音 / 一句话 / 滑杆 三选一）
    if inp.emotion_ref.strip():
        kw["emotion_reference_audio"] = inp.emotion_ref.strip()
    elif inp.emotion_text.strip():
        kw["use_emo_text"] = True
        kw["emo_text"] = inp.emotion_text.strip()
    elif inp.emotion.strip():
        kw["emotion"] = inp.emotion.strip()
        kw["emo_alpha"] = max(0.0, min(1.0, inp.emo_alpha))
    t0 = time.time()
    with LOCK:
        wav = ENGINE.generate(inp.text, inp.ref, **kw)
    dur = len(wav) / ENGINE.sample_rate
    name = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
    path = os.path.join(OUT, name)
    sf.write(path, wav, ENGINE.sample_rate)
    return {"file": name, "seconds": round(dur, 2),
            "elapsed": round(time.time() - t0, 1)}


@app.get("/audio/{name}")
def audio(name: str):
    p = os.path.join(OUT, os.path.basename(name))
    if not os.path.exists(p):
        return JSONResponse({"error": "没有这个文件"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7866, log_level="warning")
