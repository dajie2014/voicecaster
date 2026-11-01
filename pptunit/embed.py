"""把音频 + 大播放按钮嵌进幻灯片。

要害（2026-10-30 踩过）：光有 p14:media 不够，PowerPoint 会把它当普通图片，
点了没反应。必须再写一段 <p:timing>，里面有一条 mediacall 的点击动画
（presetClass="mediacall" + playFrom(0.0)），放映时点它才出声。

结构照 PowerPoint 自己插音频时的写法：
  p:pic/p:nvPicPr/p:nvPr
    <a:audioFile r:link="音频关系"/>
    <p14:media r:embed="媒体关系"/>
"""
import os
from lxml import etree
from PIL import Image, ImageDraw

from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn, nsdecls
from pptx.oxml import parse_xml

P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
etree.register_namespace("p14", P14)
CT = {".wav": "audio/wav", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".aac": "audio/mp4"}


def make_icon(path, size=512, color=(22, 30, 46, 235), ring=(120, 190, 255, 220)):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, size - 10, size - 10], radius=90,
                        fill=color, outline=ring, width=6)
    cx, cy, s = size * 0.53, size * 0.5, size * 0.20
    d.polygon([(cx - s * 0.8, cy - s), (cx - s * 0.8, cy + s), (cx + s * 1.15, cy)],
              fill=(255, 255, 255, 245))
    img.save(path)


def _timing_xml(spid, autoplay=False):
    """autoplay=True 时，翻到这一页就开始念（不依赖任何点击）。"""
    ns = nsdecls("p", "a")
    start_cond = '<p:cond delay="0"/>' if autoplay else '<p:cond delay="indefinite"/>'
    click_kind = "withEffect" if autoplay else "clickEffect"
    return (
        f'<p:timing {ns}>'
        '<p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        '<p:childTnLst><p:seq concurrent="1" nextAc="seek">'
        '<p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
        '<p:par><p:cTn id="3" fill="hold"><p:stCondLst>' + start_cond + '</p:stCondLst>'
        '<p:childTnLst><p:par><p:cTn id="4" fill="hold">'
        '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
        '<p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" '
        'fill="hold" nodeType="' + click_kind + '"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
        '<p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr>'
        '<p:cTn id="6" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="' + str(spid) + '"/>'
        '</p:tgtEl></p:cBhvr></p:cmd>'
        '<p:cMediaNode vol="80000" numSld="1" showWhenStopped="0">'
        '<p:cTn id="7" fill="hold" display="0"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
        '<p:tgtEl><p:spTgt spid="' + str(spid) + '"/></p:tgtEl>'
        '</p:cMediaNode></p:childTnLst></p:cTn></p:par>'
        '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
        '</p:childTnLst></p:cTn>'
        '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
        '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
        '</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'
    )


def add_timing(slide, spid, autoplay=False):
    """在页面末尾写播放时序；同一页已有 timing 就替换。"""
    sld = slide._element
    old = sld.find(qn("p:timing"))
    if old is not None:
        sld.remove(old)
    sld.append(parse_xml(_timing_xml(spid, autoplay)))


def strip_audio(slide):
    """重做一页之前，把这一页上旧的音频按钮清掉，不然会叠两层。"""
    removed = 0
    for sh in list(slide.shapes):
        try:
            if (sh.name or "").startswith("audio_"):
                sh._element.getparent().remove(sh._element)
                removed += 1
        except Exception:
            pass
    sld = slide._element
    old = sld.find(qn("p:timing"))
    if old is not None:
        sld.remove(old)
    return removed


def add_audio(slide, audio_path, icon_path, left, top, width, height, tag,
              descr="点击播放这一页的讲解", autoplay=False):
    """用 python-pptx 官方的 add_movie 插音频。

    2026-10-30 的教训：手工拼 p:pic + audioFile + p14:media 看起来对，
    PowerPoint 也认成音频对象，但点了不播。官方接口会额外生成 p:video
    动画节点（`_add_video_timing`），那是播放控件真正认的东西。
    """
    ext = os.path.splitext(audio_path)[1].lower()
    mime = CT.get(ext, "audio/wav")
    pic = slide.shapes.add_movie(str(audio_path), left, top, width, height,
                                 poster_frame_image=icon_path, mime_type=mime)
    pic.name = f"audio_{tag}"
    try:
        pic._element.nvPicPr.cNvPr.set("descr", descr)
    except Exception:
        pass
    return pic


def embed_slide(slide, audio_path, icon_path, slide_w, slide_h, tag, autoplay=False):
    """右下角放一个大按钮（约占幻灯片高度 20%）。"""
    h = int(slide_h * 0.20)
    w = h
    left = int(slide_w - w - slide_h * 0.06)
    top = int(slide_h - h - slide_h * 0.05)
    return add_audio(slide, audio_path, icon_path, left, top, w, h, tag, autoplay=autoplay)
