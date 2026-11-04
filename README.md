# VoiceCaster

**捏嗓子、念剧本、做有声课件。** 引擎分工（2026-11-04 定）：
**中文的讲课/朗读和演戏 → 千问（小户）**，演戏里的方言交给 **Vox（大户）**，同场联合；
**小米 OmniVoice（中户）只接外文稿**，平时闲着；IndexTTS2.5（戏骨）负责更细的情绪，按需起。

- **怎么用（日常操作，看这个就够）**：`使用说明.md`
- 需求（要做什么）：`SPEC.md`
- 进度与坑（做到哪、踩过什么）：`ROADMAP.md`

---

## 怎么开

双击 `/Applications/VoiceCaster.app`。它会：

1. 起 `app.py`（`:7863`，**旧界面，现在可以不用**——捏角色已搬进主界面）
2. 起 `server.py`（`:7864`，主界面，自己写的网页）
3. 打开 `http://127.0.0.1:7864`

手动起主服务（调试时用）：

```sh
cd ~/Documents/deepseek-harness/default-workspace/voicecaster
env HF_HOME=$HOME/Documents/deepseek-harness/default-workspace/voxcpm-tentacle/.hf \
    HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    ../voxcpm-tentacle/.venv/bin/python -u server.py
```

模型加载约 10 秒。**起一个就够**，两个模型同时驻留会各占 6 GB。

---

## 四个页面

| 标签 | 干什么 |
|---|---|
| **角色** | 看角色、试听、念一句、**写描述捏新嗓子（一次三条候选）**、删 |
| **剧本** | 贴剧本 → 读剧本（角色自动对号）→ 排演（整段/分场）→ 单句重念 → 打点 → 导出包 |
| **PPT 讲解** | 导入 .pptx → 抽备注拆句（可改可删）→ 选声音、选格式 → 生成 → 嵌回带播放键 → 下载课件/音频包 |
| **声音克隆** | 选声卡、录音（10 秒自动停、电平表）、试听、存进克隆库（改名/删） |

---

## 代码结构

```
server.py           主服务：角色/剧本/排演/导出/PPT/克隆 全部接口
app.py              :7863 旧界面（Gradio），捏角色已搬进主界面，留作备份
parser.py           剧本解析：角色：台词（情绪）【音效】
motion.py           动作提示 → 声音描述（本机 Ollama 小模型）
web/index.html      主界面（单文件，HTML+CSS+JS 全在里面）
pptunit/            PPT 教学单元
  notes.py          抽每页备注（没有就退到正文）
  split.py          拆成 2–3 句能单播的口语（去导航句、不断「而是」）
  glossary.py       术语表：整词命中 + 断句边界
  embed.py          音频 + 大按钮 + 点击播放的 timing
  build.py          CLI 端到端（不经过网页也能跑一册）
  verify.py         解包检查每页 rels 是否正常
tests/smoke.py      冒烟测试：7 条接口，几秒钟跑完
tools/precheck_indextts.py   装 IndexTTS 前的环境体检（uv/外盘空间/网络，不下载）
characters/         角色卡（N.json + N.wav）
output/ppt/         每册课件的音频、manifest、成品 pptx
output/clone/       克隆用的参考音频与试听
projects/           保存的剧本工程
```

---

## 改完先跑这个

```sh
../voxcpm-tentacle/.venv/bin/python tests/smoke.py    # 服务要先开着
```

7 条接口，覆盖角色、解析、存读、术语、PPT、导出、克隆。**生成音频慢，不在里面。**

---

## 两条硬规矩（改代码时别破坏）

1. **PPT 教学只走 VoxCPM2**，永远不调 IndexTTS（省得为了念一页课件去加载外盘大模型）。
2. **不做视频编码 / 音画对齐 / 闭源格式**。对齐是学生在 CapCut 里做的事。
