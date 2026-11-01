#!/bin/bash
# VoiceCaster 启动器（Apple 芯片 Mac）
#
# 用法：
#   1. 把 VOICE_ROOT 指到你放 Vox 那套（含 .venv 和 .hf）的目录
#   2. 把 QWEN_ROOT 指到千问 TTS 服务的目录
#   3. sh run-mac.sh   （或做成 .app 双击）
#
# 它只做三件事：缺哪个服务就拉起来、等它就绪、打开界面。

DIR="$(cd "$(dirname "$0")" && pwd)"

# ⚠️ 这两个是本地路径，按你的机器改（也可以用环境变量覆盖）
VOICE_ROOT="${VOICE_ROOT:-$DIR/../voxcpm-tentacle}"   # 含 .venv / .hf 的那套
QWEN_ROOT="${QWEN_ROOT:-$DIR/../qwen-tts}"            # 千问 TTS 常驻服务

VENV="$VOICE_ROOT/.venv"
QVENV="$QWEN_ROOT/.venv"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$VOICE_ROOT/.hf}"
export HF_HUB_DISABLE_XET=1

cd "$DIR" || exit 1

# 千问语音（:3910）：试听、朗读、剧本都靠它念；没有它一律报 503
if ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:3910/health"; then
  if [ -x "$QVENV/bin/python" ]; then
    ( cd "$QWEN_ROOT" && nohup "$QVENV/bin/python" tts_server.py > /tmp/qwen-tts.log 2>&1 & )
  else
    echo "⚠️  没找到千问服务：$QVENV/bin/python"
    echo "    先装好它，或自己起一个兼容 POST /tts 的服务（见 README 第一节）"
  fi
fi

# 主服务（:7864）
if ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:7864/api/characters"; then
  nohup "$VENV/bin/python" server.py > /tmp/voicecaster-main.log 2>&1 &
fi

osascript -e 'display notification "第一次启动要等半分钟，页面会自己打开" with title "VoiceCaster 正在启动"' 2>/dev/null
for i in $(seq 1 150); do
  sleep 1
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:7864/"; then
    open "http://127.0.0.1:7864/"
    exit 0
  fi
done
echo "起不来，看 /tmp/voicecaster-main.log 最后几行"
