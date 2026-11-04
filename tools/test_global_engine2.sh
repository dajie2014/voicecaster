#!/bin/bash
# 官方音色 + 全局强制 Vox：应当自动退回千问，不能崩
set -u
VC=~/Documents/deepseek-harness/default-workspace/voicecaster
T=~/Documents/deepseek-harness/default-workspace/voxcpm-tentacle
cd "$VC" || exit 1

HF_HOME=$T/.hf TORCH_HOME=$T/.torch HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  nohup "$T/.venv/bin/python" server.py > /tmp/vc-test2.log 2>&1 &
PID=$!
for i in $(seq 1 150); do
  sleep 1
  curl -s -m 2 -o /dev/null http://127.0.0.1:7864/api/characters && break
done

echo "== 首页里有下拉吗 =="
curl -s http://127.0.0.1:7864/ | grep -c "id=\"globalEngine\""
echo "== 全局设 Vox =="
curl -s -X POST -H 'Content-Type: application/json' -d '{"engine":"voxcpm"}' \
  http://127.0.0.1:7864/api/settings/engine; echo
echo "== 官方音色卡(2) 念一句：应退回千问 =="
curl -s -m 120 -X POST -H 'Content-Type: application/json' \
  -d '{"text":"官方音色没有参考音。"}' http://127.0.0.1:7864/api/characters/2/say; echo
echo "== 恢复 follow =="
curl -s -X POST -H 'Content-Type: application/json' -d '{"engine":"follow"}' \
  http://127.0.0.1:7864/api/settings/engine; echo

kill "$PID" 2>/dev/null
sleep 2
echo "已关"
