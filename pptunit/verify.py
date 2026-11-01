"""检查一份「有声课件」是不是真的把音频嵌进去了。

不靠 PowerPoint 的肉眼判断，直接解包看三处：
1. 每页 rels 里有没有 audio 和 media 两条关系
2. 它们指向的媒体文件在包里吗
3. 页面 XML 里有没有 p14:media 引用

用法：python verify.py 课件_有声.pptx
"""
import re
import sys
import zipfile

AUDIO = "relationships/audio"
MEDIA = "relationships/media"


def verify(path):
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[0]))
    ok = bad = 0
    for s in slides:
        idx = re.findall(r"\d+", s)[0]
        rel = f"ppt/slides/_rels/slide{idx}.xml.rels"
        if rel not in names:
            print(f"  第 {idx} 页：缺 rels"); bad += 1; continue
        r = z.read(rel).decode("utf-8", "ignore")
        has_audio = AUDIO in r
        has_media = MEDIA in r
        xml = z.read(s).decode("utf-8", "ignore")
        # 两种合法写法都算：手工 p14:media，或 python-pptx 官方 add_movie 的 p:video
        has_p14 = (("powerpoint/2010/main" in xml and "media" in xml)
                   or "<p:video" in xml or "audioFile" in xml or "videoFile" in xml)
        targets = re.findall(r'Target="\.\./media/([^"]+)"', r)
        exist = all(f"ppt/media/{t}" in names for t in targets) if targets else False
        flag = "OK" if ((has_audio or has_media) and has_p14 and exist and targets) else "有问题"
        if flag == "OK":
            ok += 1
        else:
            bad += 1
        print(f"  第 {idx} 页：audio={has_audio} media={has_media} "
              f"p14={has_p14} 文件在={exist} {targets} → {flag}")
    print(f"\n{len(slides)} 页：{ok} 页正常，{bad} 页有问题")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if verify(sys.argv[1]) else 1)
