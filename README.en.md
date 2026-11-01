# VoiceCaster

**A local-first Chinese voice workshop**: voice library → narration → multi-character script rehearsal → narrated slides.
Everything runs on your own machine. No network, no per-character billing.

Built for a composer's own teaching and radio-drama workflow: turn slide notes into narration,
turn a script into a multi-voice scene, and tweak the delivery of individual sentences.

> This is a **personal workflow tool**, not a shrink-wrapped product. The first section below says
> exactly what runs as-is, what needs porting, and what has to be rewritten — so you don't waste a day.

> ⚠️ **One thing to know up front: the voice instructions are written in Chinese.**
> Expressions like 「压低声音，像怕人听见」(lower your voice, as if afraid of being overheard)
> are passed to the TTS model as natural-language prompts. English prompts may be ignored or
> degrade the result. The UI is Chinese-only too.

---

## Documents

| File | What it is |
|---|---|
| **使用手册.md** | The user manual, organised by *what you want to do*, not by code layout (Chinese) |
| **CHANGELOG.md** | Every change since day one, by date (Chinese) |
| **分工.md** | Who does what — human vs. AI — and the line that must not move (Chinese) |
| **SPEC.md** | The original brief, kept for design intent (Chinese) |

---

## 1. Platforms: runs / needs porting / must be rewritten

| Part | Apple-silicon Mac | Windows / Intel Mac |
|---|---|---|
| Web UI, script parsing, PPT extraction & embedding, export | ✅ works as-is | ✅ works as-is (pure Python) |
| Speech: Qwen TTS (the nine official voices) | ✅ local MLX | 🔧 needs a PyTorch implementation |
| Speech: Vox (voice design, arbitrary dialects) | ✅ local MLX | ✍️ rewrite or drop |

### ✅ Works as-is (chip-agnostic)

- `parser.py` — script parsing (`name: line`)
- `pptunit/` — extract speaker notes, split sentences, embed audio back into the pptx, export
- `web/index.html` — the whole UI
- `tests/smoke.py` — smoke test

### 🔧 Needs porting (swap the speech backend; nothing above it changes)

Three places are hard-wired to Apple silicon:

1. **`server.py`, top of file**
   `MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", ..., device="mps")`
   — bound to Apple MPS/MLX. Replace with the PyTorch/CUDA build of VoxCPM.
2. **The Qwen TTS sidecar service** (not in this repo)
   The current implementation uses `mlx_audio` and listens on `http://127.0.0.1:3910`.
   On Windows, swap in the PyTorch build of Qwen3-TTS and **keep the same HTTP contract**:
   `POST /tts {text, voice, lang, instruct, speed}` → wav.
3. **`indextts_server.py`** — uses `mlx_indextts`; same story.

> The good news: nothing above the speech layer knows who is speaking. Every synthesis call in
> `server.py` funnels through one HTTP call (`_qwen_wav()`). **Keep that contract and the script,
> narration and PPT logic needs zero changes.**

### ✍️ Rewrite or drop

- **Dialects**: 「say it in Sichuan dialect」 is currently delegated to Vox.
  Without Vox on Windows you lose it — only Qwen's built-in Sichuan (eric) and Beijing (dylan) remain.
- **Performance**: without an NVIDIA GPU, CPU inference takes tens of seconds per sentence.
- **ffmpeg path**: hard-coded to `/opt/homebrew/bin/ffmpeg` in `pptunit/embed.py` — change it.

---

## 2. Models (not in the repo — download them yourself)

Five models, **≈14.7 GB** in total. For narration only you need the first one (2.9 GB).

| # | Repository | Size | Purpose | Required? |
|---|---|---|---|---|
| 1 | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 2.9 G | The nine official voices: preview, narration, script, slides | **yes** |
| 2 | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | 2.9 G | Clone your own voice | for cloning |
| 3 | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | 2.9 G | Design a new voice from a text description | for voice design |
| 4 | `openbmb/VoxCPM2` | 4.6 G | Voice design, acting, **dialects** | for dialects/acting |
| 5 | `deepdml/faster-whisper-large-v3-turbo-ct2` | 1.5 G | Transcribe the reference clip (cloning won't work without it) | for cloning |

> Sizes are disk usage, not memory. Free up 15 GB before downloading.
> In mainland China: `export HF_ENDPOINT=https://hf-mirror.com`.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    local_dir="models/qwen3tts",        # -Base / -VoiceDesign for the other two
)
```

**Memory (rough, not measured per-item)**: Qwen alone ≈3–4 GB; Vox alone ≈6 GB;
both at once ≈10 GB+, which is tight on a 16 GB machine. `run-mac.sh` starts what you need and
lets it go when idle.

---

## 3. Run it (Apple-silicon Mac)

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# start the speech sidecar first (installed separately, default http://127.0.0.1:3910)
python server.py          # → http://127.0.0.1:7864
```

The main service loads VoxCPM2 at startup (~10 s). One model resident is enough —
two at once will each take ~6 GB.

`run-mac.sh` is the launcher: checks ports, starts what is missing, opens the UI.

---

## 4. The five pages

| Tab | What it does |
|---|---|
| **声库** Voice library | The nine official voices + your own (designed / cloned / imported). Preview, rename, set dialect, replace photo |
| **剧本** Script | Paste → parse (names matched to voices automatically) → rehearse → redo a single line → export |
| **朗读** Narration | Paste text → pick narrator/character → generate |
| **PPT 讲解** Slides | Import `.pptx` → extract speaker notes → split → generate → embed back with play buttons → download |
| **声音克隆** Cloning | Pick input device, record (auto-stop at 10 s, level meter), preview, save |

Two things are worth knowing because they took most of the work:

- **Select a few characters inside a line and give them their own delivery** (natural language,
  e.g. 「压低声音，像怕人听见」); **insert a pause at the caret**. Both the narration and script
  pages support this.
- It is measurable, not vibes: the same sentence with 「气声」(breathy) runs 16 % longer,
  18 % quieter and 41 Hz lower; with 「高兴」(happy) it runs 24 % faster and 70 Hz higher.
  The same instruction, run three times, gives identical output.

---

## 5. Layout

```
server.py          main service (FastAPI, :7864)
parser.py          script parser
dialects.py        dialect detection
motion.py          stage-direction → voice description
pptunit/           PPT: notes, sentence splitting, audio embedding, verification
web/index.html     the whole UI (vanilla JS, no framework)
portraits/         the nine voice portraits (samples)
tools/ tests/      small utilities and a smoke test
```

---

## 6. License

MIT. Do whatever you want; don't come looking for me if it breaks.
