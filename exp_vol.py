import os, torch, soundfile as sf
from voxcpm import VoxCPM
BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters"); OUT = os.path.join(BASE, "exp_vol")
os.makedirs(OUT, exist_ok=True)
M = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
SR = M.tts_model.sample_rate
REF = os.path.join(CHARS, "1.wav")
LINE = "这次的任务，先核对截图信息，整理汇总。"
RUNS = {
  "1_正常":  "中年男人，中音，正常说话",
  "2_很小声": "中年男人，中音，声音很小，几乎听不见，气声，贴着话筒说",
  "3_很大声": "中年男人，中音，音量很大，放大声音说，在房间里都听得见",
}
for name, ctrl in RUNS.items():
    torch.manual_seed(20261029)
    wav = M.generate(text=f"({ctrl}){LINE}", reference_wav_path=REF, cfg_value=2.2, inference_timesteps=16)
    sf.write(os.path.join(OUT, f"{name}.wav"), wav, SR)
    print(f"{name}: {len(wav)/SR:.2f}s", flush=True)
