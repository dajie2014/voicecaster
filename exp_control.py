"""测：模型的 control 对「情绪描述」和「动作描述」分别有没有反应"""
import os, time, torch, soundfile as sf
from voxcpm import VoxCPM

BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters")
OUT = os.path.join(BASE, "exp")
os.makedirs(OUT, exist_ok=True)

M = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
SR = M.tts_model.sample_rate
REF = os.path.join(CHARS, "1.wav")
LINE = "我明白你的顾虑。但能省就省。"

GROUPS = {
    "A_情绪": "中年男人，中音，声音压得很低，说得很慢，几乎是在叹气",
    "B_动作": "他站在办公室里，一只手撑着桌沿，另一只手指着屏幕，一动不动",
    "C_空":   "中年男人，中音",
}

for name, ctrl in GROUPS.items():
    torch.manual_seed(20261029)
    wav = M.generate(text=f"({ctrl}){LINE}", reference_wav_path=REF,
                     cfg_value=2.2, inference_timesteps=16)
    p = os.path.join(OUT, f"{name}.wav")
    sf.write(p, wav, SR)
    print(f"{name}: {len(wav)/SR:.2f}s", flush=True)

print("done", flush=True)
