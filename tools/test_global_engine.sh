#!/bin/bash
# 起一遍 VoiceCaster，验全局引擎下拉的接口与分发，然后关掉
set -u
VC=~/Documents/deepseek-harness/default-workspace/voicecaster
T=~/Documents/deepseek-harness/default-workspace/voxcpm-tentacle
cd "$VC" || exit 1

if curl -s -m 2 -o /dev/null http://127.0.0.1:7864/api/characters; then
  echo "已有服务在跑，直接测（测完不关它）"; PID=""
else
  HF_HOME=$T/.hf TORCH_HOME=$T/.torch HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    nohup "$T/.venv/bin/python" server.py > /tmp/vc-test.log 2>&1 &
  PID=$!
  for i in $(seq 1 150); do
    sleep 1
    curl -s -m 2 -o /dev/null http://127.0.0.1:7864/api/characters && break
  done
fi

echo "== 初始 =="; curl -s http://127.0.0.1:7864/api/settings/engine; echo
echo "== 设成小米 =="; curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"engine":"omnivoice"}' http://127.0.0.1:7864/api/settings/engine; echo
echo "== 念一句（应走小米） =="
curl -s -m 120 -X POST -H 'Content-Type: application/json' \
  -d '{"text":"这一句用来验证全局下拉。"}' http://127.0.0.1:7864/api/characters/1/say; echo
echo "== 设成千问 =="; curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"engine":"qwen"}' http://127.0.0.1:7864/api/settings/engine; echo
echo "== 念一句（应走千问） =="
curl -s -m 120 -X POST -H 'Content-Type: application/json' \
  -d '{"text":"这一句用来验证全局下拉。"}' http://127.0.0.1:7864/api/characters/1/say; echo
echo "== 恢复跟随角色卡 =="; curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"engine":"follow"}' http://127.0.0.1:7864/api/settings/engine; echo
echo "== 非法值（应回落 follow） =="; curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"engine":"nonsense"}' http://127.0.0.1:7864/api/settings/engine; echo

if [ -n "$PID" ]; then
  kill "$PID" 2>/dev/null
  sleep 2
  echo "已关闭测试服务（pid $PID）"
fi
