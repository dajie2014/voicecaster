"""VoiceCaster 冒烟测试：不起模型，只打接口，几秒钟跑完。

用途：改完代码先跑这个，确认没有哪条路被改断。
    python tests/smoke.py            # 默认测 http://127.0.0.1:7864
    python tests/smoke.py --port 7864

不测「生成音频」（要跑模型、几分钟）；那种用 test_deck.pptx 真跑一页来验。
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.parse
import zipfile
import io

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = "http://127.0.0.1"


def call(method, path, body=None, timeout=30):
    url = HOST + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        ctype = r.headers.get("Content-Type", "")
        if "json" in ctype:
            return r.status, json.loads(raw)
        return r.status, raw


def main():
    global HOST
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7864)
    a = ap.parse_args()
    HOST = "http://127.0.0.1:" + str(a.port)

    ok, bad = 0, 0

    def check(name, fn):
        nonlocal ok, bad
        try:
            msg = fn()
            print(f"  ✅ {name}" + (f" — {msg}" if msg else ""))
            ok += 1
        except Exception as e:
            print(f"  ❌ {name} — {type(e).__name__}: {e}")
            bad += 1

    print("VoiceCaster 冒烟测试", HOST)

    def chars():
        s, d = call("GET", "/api/characters")
        assert s == 200 and isinstance(d, list), "角色列表不对"
        return f"{len(d)} 个角色"

    TMP_CHAR = {"id": None}

    def ensure_char(name="阿达"):
        """库里没有这个角色就造一个临时的，测完删掉。

        以前这条断言硬绑「1 号角色」——库一清空就挂，而且挂得跟代码无关
        （2026-10-30 就因为清了库，这条报「名字没对上 1 号」）。
        """
        s, d = call("GET", "/api/characters")
        for c in d:
            if c.get("name") == name:
                return c["id"]
        p = os.path.join(BASE, "output", "clone", "upload_48k.wav")
        if not os.path.exists(p):
            return None
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        s, d = call("POST", "/api/clone/upload", {"filename": "smoke.wav", "data": b64}, timeout=60)
        s, d = call("POST", "/api/clone/save", {"ref": d["ref"], "name": name})
        TMP_CHAR["id"] = d["card"]["id"]
        return d["card"]["id"]

    def parse():
        cid = ensure_char()
        if cid is None:
            return "跳过（没有现成样本，造不出临时角色）"
        s, d = call("POST", "/api/parse", {"text": "阿达：这次的任务。\n小迪：明白。"})
        assert s == 200 and len(d["lines"]) == 2, "解析结果不对"
        assert d["lines"][0]["char_id"] == cid, f"名字没对上 {cid} 号"
        return f"{len(d['lines'])} 句，summary {len(d.get('summary', []))} 条"

    def project():
        s, d = call("POST", "/api/project/save",
                    {"name": "冒烟测试项目", "script": "阿达：一。", "voices": {}})
        assert s == 200 and d.get("ok"), "保存失败"
        s, d = call("GET", "/api/project/" + urllib.parse.quote("冒烟测试项目"))
        assert d.get("script") == "阿达：一。", "读回来的内容不对"
        return "存 → 读 一致"

    def glossary():
        s, d = call("GET", "/api/ppt/glossary")
        assert s == 200 and "terms" in d, "术语表读不到"
        return f"{len(d['terms'])} 条术语"

    def ppt_analyze():
        p = os.path.join(BASE, "pptunit", "test_deck.pptx")
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        s, d = call("POST", "/api/ppt/analyze", {"name": "冒烟测试", "data": b64}, timeout=60)
        assert s == 200 and len(d["pages"]) >= 1, "课件读不出页"
        assert any(pg["sentences"] for pg in d["pages"]), "一页都没拆出句子"
        return f"{len(d['pages'])} 页"

    def export():
        s, d = call("POST", "/api/export/package", {
            "name": "冒烟测试包",
            "files": ["001_1号_阿达.wav"],
            "lines": [{"no": 1, "name": "阿达", "text": "这次的任务。"}],
            "format": "m4a", "srt": True, "marks": {"1": 3.5}}, timeout=120)
        assert s == 200 and d.get("count") == 1, "导出没成功"
        s, raw = call("GET", urllib.parse.quote(d["url"], safe="/"), timeout=60)
        z = zipfile.ZipFile(io.BytesIO(raw))
        names = z.namelist()
        assert any(n.endswith(".m4a") for n in names), "包里没有音频"
        assert "Manifest.txt" in names or any("Manifest" in n for n in names), "包里没有清单"
        return f"{len(names)} 个文件"

    def clone_upload():
        p = os.path.join(BASE, "output", "clone", "upload_48k.wav")
        if not os.path.exists(p):
            return "跳过（没有现成样本）"
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        s, d = call("POST", "/api/clone/upload", {"filename": "x.wav", "data": b64}, timeout=60)
        assert s == 200 and d["seconds"] > 0, "上传/转码失败"
        return f"{d['seconds']} 秒"

    check("角色列表", chars)
    check("剧本解析 + 角色匹配", parse)
    check("项目存读", project)
    check("术语表", glossary)
    check("PPT 读课件", ppt_analyze)
    check("导出包", export)
    check("克隆上传", clone_upload)

    if TMP_CHAR["id"]:
        try:
            call("POST", f"/api/characters/{TMP_CHAR['id']}/delete")
            print(f"  （顺手删掉测试造的临时角色 {TMP_CHAR['id']} 号）")
        except Exception:
            pass

    print(f"\n{ok} 过，{bad} 挂")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
