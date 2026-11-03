"""原型：往 pptx 每页嵌一段音频 + 一个大播放按钮。

只验证一件事：PowerPoint 打开时不报"需要修复"，并且页面上出现可点的音频对象。
先不接 VoxCPM，用现成的 wav 当占位。
"""
import os, sys
from lxml import etree
from PIL import Image, ImageDraw

from pptx import Presentation
from pptx.util import Emu, Inches, Pt
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn

P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
HERE = os.path.dirname(os.path.abspath(__file__))


def make_icon(path, size=512):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, size - 10, size - 10], radius=90,
                        fill=(22, 30, 46, 235), outline=(120, 190, 255, 220), width=6)
    cx, cy, s = size * 0.53, size * 0.5, size * 0.20
    d.polygon([(cx - s * 0.8, cy - s), (cx - s * 0.8, cy + s), (cx + s * 1.15, cy)],
              fill=(255, 255, 255, 245))
    img.save(path)


def add_audio(slide, wav_path, icon_path, left, top, width, height, tag):
    pkg = slide.part.package
    ext = os.path.splitext(wav_path)[1]
    ct = "audio/wav" if ext == ".wav" else "audio/mp4"
    used = {str(p.partname) for p in pkg.iter_parts()}
    n = 0
    while True:
        pn = PackURI(f"/ppt/media/{tag}{'' if n == 0 else n}{ext}")
        if str(pn) not in used:
            break
        n += 1
    part = Part(pn, ct, pkg, open(wav_path, "rb").read())
    r_media = slide.part.relate_to(part, RT.MEDIA)
    r_audio = slide.part.relate_to(part, RT.AUDIO)

    pic = slide.shapes.add_picture(icon_path, left, top, width, height)
    pic.name = f"audio_{tag}"
    nvPicPr = pic._element.nvPicPr
    nvPicPr.cNvPr.set("descr", "点击播放这一页的讲解")
    nvPr = nvPicPr.find(qn("p:nvPr"))
    if nvPr is None:
        nvPr = etree.SubElement(nvPicPr, qn("p:nvPr"))
    af = etree.SubElement(nvPr, qn("a:audioFile"))
    af.set(qn("r:link"), r_audio)
    med = etree.SubElement(nvPr, "{%s}media" % P14)
    med.set(qn("r:embed"), r_media)
    return pic


def make_test_deck(path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    pages = [
        ("第一页 · 为什么要用这个", "各位同学，今天我们讲声音里的「距离感」。"
                                  "很多人以为换个音量就够了，其实高频的衰减才是关键。"),
        ("第二页 · 三个参数", "第一个参数是高频衰减。"
                            "第二个是直达声和混响的比例。第三个是起音的速度。"),
        ("第三页 · 现场演示", "现在我们听一段对比。"
                           "注意听第二遍，同一句话，但距离感完全不同。"),
    ]
    for title, notes in pages:
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.5), Inches(1.2))
        tb.text_frame.text = title
        for p in tb.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(32)
                r.font.bold = True
        s.notes_slide.notes_text_frame.text = notes
    prs.save(path)


def main():
    icon = os.path.join(HERE, "play_icon.png")
    make_icon(icon)
    deck = os.path.join(HERE, "test_deck.pptx")
    out = os.path.join(HERE, "test_deck_audio.pptx")
    make_test_deck(deck)
    wavs = [os.path.join(HERE, "..", "output", "001_1号_阿达.wav"),
            os.path.join(HERE, "..", "output", "002_2号_小迪.wav"),
            os.path.join(HERE, "..", "output", "001_1号_阿达.wav")]
    prs = Presentation(deck)
    for i, slide in enumerate(prs.slides):
        add_audio(slide, os.path.abspath(wavs[i]), icon,
                  Inches(10.8), Inches(5.6), Inches(1.5), Inches(1.5), f"slide{i+1}")
    prs.save(out)
    print("saved", out, os.path.getsize(out))


if __name__ == "__main__":
    main()
