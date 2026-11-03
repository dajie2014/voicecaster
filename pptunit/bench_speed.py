"""测一次真实速度：加载 + 三档步数。数字出来后决定 PPT 单元怎么跑。"""
import time, os
from voxcpm import VoxCPM
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
print("load %.1fs" % (time.time() - t0), flush=True)
ref = os.path.join(BASE, "characters", "1.wav")
sr = m.tts_model.sample_rate
text = "各位同学，今天我们讲声音里的距离感。很多人以为换个音量就够了。"
for steps in (8, 16, 32):
    t = time.time()
    w = m.generate(text=text, reference_wav_path=ref, cfg_value=2.2, inference_timesteps=steps)
    print("steps=%d  %.2fs  audio=%.2fs" % (steps, time.time() - t, len(w) / sr), flush=True)
