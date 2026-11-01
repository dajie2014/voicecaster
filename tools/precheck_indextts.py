"""装 IndexTTS2.5-MLX 之前，先看这台机器够不够。

只检查，不下载、不安装。
    ../voxcpm-tentacle/.venv/bin/python tools/precheck_indextts.py
"""
import os
import shutil
import subprocess
import sys


def run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def main():
    print("== IndexTTS2.5-MLX 安装前检查 ==\n")
    ok = True

    # 1) uv
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if os.path.exists(uv):
        r = run([uv, "--version"])
        print(f"  工具 uv：✅ {r.stdout.strip() if r else ''}")
    else:
        print("  工具 uv：❌ 没找到（官方要求用它管理依赖）")
        ok = False

    # 2) 磁盘
    for vol in ("/Volumes/x", "/Volumes/REBIRTH"):
        if os.path.exists(vol):
            try:
                st = os.statvfs(vol)
                free = st.f_bavail * st.f_frsize / 1e9
                mark = "✅" if free > 20 else "⚠️"
                print(f"  外盘 {vol}：{mark} 剩 {free:.0f} GB（权重+转换产物，估算需要 10–20 GB）")
            except Exception as e:
                print(f"  外盘 {vol}：读不到（{e}）")
        else:
            print(f"  外盘 {vol}：不在")

    # 3) mlx
    try:
        import mlx.core  # noqa: F401
        print("  MLX 框架：✅ 已装")
    except Exception:
        print("  MLX 框架：⬜ 没装（`uv sync --extra v25` 会装）")

    # 4) 网络（hf-mirror）
    r = run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10",
             "https://hf-mirror.com"])
    code = r.stdout.strip() if r else "超时"
    good = code in ("200", "301", "302")
    print(f"  hf-mirror：{'✅' if good else '❌'} {code}")

    # 5) 现有 VoxCPM 是否还在
    venv = os.path.expanduser("~/Documents/deepseek-harness/default-workspace/voxcpm-tentacle/.venv")
    print(f"  VoxCPM venv：{'✅' if os.path.exists(venv) else '❌'} {venv}")

    print("\n结论：", "可以装。" if ok else "先补上面标 ❌ 的东西。")
    print("装法写在 SPEC.md 的「装法」那一节。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
