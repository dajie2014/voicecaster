import os, torch, soundfile as sf
from voxcpm import VoxCPM
BASE = os.path.dirname(os.path.abspath(__file__))
CHARS = os.path.join(BASE, "characters"); OUT = os.path.join(BASE, "exp")
M = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
SR = M.tts_model.sample_rate
REF = os.path.join(CHARS, "1.wav")
LINE = "我明白你的顾虑。但能省就省。"
RUNS = {
  "C2_同提示再来一次": "中年男人，中音",
  "D_动作带声音线索": "他背对着门站着，声音闷在喉咙里，很小，像怕人听见",
}
for name, ctrl in RUNS.items():
    torch.manual_seed(777)
    wav = M.generate(text=f"({ctrl}){LINE}", reference_wav_path=REF, cfg_value=2.2, inference_timesteps=16)
    sf.write(os.path.join(OUT, f"{name}.wav"), wav, SR)
    print(f"{name}: {len(wav)/SR:.2f}s", flush=True)
