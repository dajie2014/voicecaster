import os, torch, soundfile as sf
from voxcpm import VoxCPM

OUT = "/Users/jieda/Documents/deepseek-harness/default-workspace/VoiceCaster/demo"
os.makedirs(OUT, exist_ok=True)

m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, device="mps", optimize=False)
sr = m.tts_model.sample_rate

def gen(text, name, control, ref=None):
    full = f"({control}){text}"
    wav = m.generate(text=full, reference_wav_path=ref, cfg_value=2.2, inference_timesteps=16)
    p = os.path.join(OUT, name)
    sf.write(p, wav, sr)
    print(f"{name}: {len(wav)/sr:.2f}s", flush=True)

REF = "/Users/jieda/Documents/deepseek-harness/default-workspace/voxcpm-tentacle"

# 1. 广告旁白（华丽、靠前、像电视购物）
gen("小迪插件，一切皆是插件。冷静推理，高效响应，近乎自动化的无敌工作流程，帮你搞定所有复杂任务。",
    "01_旁白.wav",
    control="flashy TV commercial announcer, bright and exaggerated, fast-paced, confident, slightly unreal, front and close")

# 2. 阿达（用已有定妆克隆；台词带派活的口吻）
gen("我明白你的顾虑。但能省就省。小奥，文字整理交给你。",
    "04_阿达.wav", ref=f"{REF}/ada_ref.wav",
    control="calm, businesslike, slightly impatient but trying to be patient")

# 3. 小奥（软、薄、犹豫、音量小）
gen("……那张图，我最开始……就想说，参数、好像不是96。",
    "05_小奥.wav",
    control="soft thin male voice, hesitant and quiet, uneven pacing, timid, struggling to speak up")

# 4. 阿碧（清亮、快、句尾上扬、自信过头）
gen("放心吧老板，没问题老板！核对完成，截图参数是96。",
    "06_阿碧.wav",
    control="bright female voice, energetic, fast, cheerful, overconfident, pitch rising at the end")

# 5. 小迪（用已有定妆克隆；平静、礼貌、居高临下）
gen("等我核一下，等我再跑一次。他们这种基础粗核对，我放心不下，容错太低。",
    "07_小迪.wav", ref=f"{REF}/di_ref.wav",
    control="even, polite, condescending, unbothered")
