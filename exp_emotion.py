import os, torch, soundfile as sf
from voxcpm import VoxCPM
BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters"); OUT = os.path.join(BASE, "exp_emo")
os.makedirs(OUT, exist_ok=True)
M = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
SR = M.tts_model.sample_rate
REF = os.path.join(CHARS, "1.wav")
LINE = "你别走。"
RUNS = {
  "1_平静":  "中年男人，中音，平静地说",
  "2_哭":    "哭腔，抽泣着，声音破碎，断断续续，几乎说不成句",
  "3_笑":    "忍不住笑出来，边笑边说，喘不上气，声音发抖",
  "4_怒吼":  "怒吼，音量很大，声音撕开，几乎是喊出来，气息很重",
}
for name, ctrl in RUNS.items():
    torch.manual_seed(20261029)
    wav = M.generate(text=f"({ctrl}){LINE}", reference_wav_path=REF, cfg_value=2.2, inference_timesteps=16)
    sf.write(os.path.join(OUT, f"{name}.wav"), wav, SR)
    print(f"{name}: {len(wav)/SR:.2f}s", flush=True)
