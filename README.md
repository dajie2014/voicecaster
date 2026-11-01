# VoiceCaster

**本地跑的中文有声内容工坊**：声库 → 朗读 → 剧本排演 → PPT 讲解。
语音全在本机生成，不联网、不按字计费。

做它是为了自己的教学与广播剧流程：把课件备注变成旁白、把剧本变成多角色对话、
把每一句的语气按需要拧一下。

> 这是一个**个人工作流工具**，不是开箱即用的商业软件。下面把「哪块直接能用、
> 哪块要移植、哪块得重写」写在最前面，省得你白折腾。

---

## 一、平台：能用 / 要移植 / 得重写

| 部分 | Apple 芯片 Mac | Windows / Intel Mac |
|---|---|---|
| 网页界面、剧本解析、PPT 拆句与嵌回、导出 | ✅ 直接能用 | ✅ 直接能用（纯 Python） |
| 语音合成：千问 TTS（官方九条嗓子） | ✅ 本机 MLX 加速 | 🔧 要换成 PyTorch 实现 |
| 语音合成：Vox（捏新嗓子、任意方言） | ✅ 本机 MLX 加速 | ✍️ 得重写或砍掉 |

### ✅ 直接能用（与芯片无关）

- `parser.py` —— 剧本解析（「名字：台词」）
- `pptunit/` —— 抽备注、拆句、把音频嵌回 pptx、导出
- `web/index.html` —— 整个界面
- `tests/smoke.py` —— 冒烟测试

### 🔧 要移植（换掉语音后端，上层不用动）

三处是硬绑定苹果的，代码里写得明明白白：

1. **`server.py` 顶部**
   `MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", ..., device="mps")`
   绑定 Apple 的 MPS/MLX。要换 PyTorch/CUDA 版本的 VoxCPM。
2. **千问 TTS 的常驻服务**（本仓库不含，需另装）
   当前实现走 `mlx_audio`，跑在 `http://127.0.0.1:3910`。
   Windows 上换成 PyTorch 版 Qwen3-TTS，**保持这个 HTTP 契约**：
   `POST /tts {text, voice, lang, instruct, speed}` → wav。
3. **`indextts_server.py`** —— 用 `mlx_indextts`，同理要换。

> 好消息：上层完全不知道底下是谁在念。`server.py` 里所有合成最后都归到
> 一个 HTTP 调用（`_qwen_wav()`）。**换引擎只要保持这个口**，剧本、朗读、PPT
> 那些逻辑一行都不用改。

### ✍️ 得重写或直接砍掉

- **方言**：现在把「用四川话说 / 用东北话说」这类活交给 Vox 念。
  Windows 上没 Vox 就一起没有，只剩千问自带的四川话（eric）和北京话（dylan）。
- **性能**：没有 NVIDIA 显卡的话，CPU 推理一句要十几秒，做整本剧本会很慢。
- **ffmpeg 路径**：`pptunit/embed.py` 里写死 `/opt/homebrew/bin/ffmpeg`，改成你自己的。

---

## 二、模型清单（不进仓库，自己下）

一共 5 个模型，**全装约 14.7 G**。只做配音的话，只需要第 1 项（2.9 G）。

| # | 仓库 | 体积 | 干什么用 | 必须吗 |
|---|---|---|---|---|
| 1 | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 2.9 G | 官方九条嗓子：试听、朗读、剧本、PPT 都靠它念 | **必须** |
| 2 | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | 2.9 G | 克隆自己的声音 | 要克隆才装 |
| 3 | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | 2.9 G | 写一段描述，捏出一个新嗓音 | 要捏才装 |
| 4 | `openbmb/VoxCPM2` | 4.6 G | 捏声、演戏、**处理方言** | 要方言/演戏才装 |
| 5 | `deepdml/faster-whisper-large-v3-turbo-ct2` | 1.5 G | 克隆时把参考录音转成文字（不给它，克隆模型不干活） | 要克隆才装 |

> 体积是磁盘占用，不是内存。下载前先腾出 15 G。

### 怎么下

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    local_dir="models/qwen3tts",          # 另外两套换成 -Base / -VoiceDesign
)
```

国内先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

### 放哪

| 模型 | 放法 |
|---|---|
| 千问三套（1/2/3） | `qwen-tts/models/` 下的 `qwen3tts`、`qwen3tts-base`、`qwen3tts-design` |
| Vox、whisper（4/5） | HF 缓存，用 `HF_HOME` 指（默认 `~/.cache/huggingface`） |

### 内存（粗估，没逐项实测）

- 单开千问：约 3–4 G
- 单开 Vox：约 6 G
- 两个同时开：10 G 起，16 G 的机器会很紧

所以 `run-mac.sh` 里的策略是：**用到哪个开哪个**，不同时驻留。

---

## 三、跑起来（Apple 芯片 Mac）

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 先起语音服务（另装，默认 http://127.0.0.1:3910），再起主服务
python server.py          # → http://127.0.0.1:7864
```

主服务启动时会加载 VoxCPM2（约 10 秒）。**一个模型驻留就够**，
两个同时开着会各占 6 G 左右。

`run-mac.sh` 是启动器：检查端口、拉起服务、打开浏览器。

---

## 四、四个页面

| 标签 | 干什么 |
|---|---|
| **声库** | 官方九条 + 自己的（捏的 / 克隆的 / 导入的）。试听、改名字、配方言、换照片 |
| **剧本** | 贴剧本 → 读剧本（角色自动对号）→ 排演 → 单句重念 → 导出包 |
| **朗读** | 投递文字 → 选旁白 / 角色 → 生成 |
| **PPT 讲解** | 导入 .pptx → 抽备注拆句 → 生成 → 嵌回带播放键 → 下载 |

两处细节值得一提，它们是这个工具最花功夫的地方：

- **选中的几个字可以单独加语气**（自然语言，如「压低声音，像怕人听见」），
  **光标处可以插停顿**。剧本和朗读两边都吃这套。
- 耳朵之外的事能测：同一句加「气声」会慢两成、轻四成、音低四十赫兹；
  加「高兴」会快四分之一、音高七十赫兹。指令是稳定的，不是碰运气。

---

## 五、目录

```
server.py          主服务（FastAPI，:7864）
parser.py          剧本解析
dialects.py        方言识别
motion.py          动作提示转声音描述
pptunit/           PPT：抽备注、拆句、嵌入音频、验证
web/index.html     整个界面（原生 JS，无框架）
portraits/         声库九张肖像（示例）
tools/ tests/      小工具与冒烟测试
```

---

## 六、许可

MIT。怎么改都行，出问题别找我。
