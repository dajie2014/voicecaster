#!/usr/bin/env python3
"""把 Markdown 转成能打印的中文 HTML，再交给 Chrome 打成 PDF。"""
import re, html, sys, os, subprocess

def inline(t):
    t = html.escape(t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    return t

def convert(md):
    lines = md.split('\n')
    out, i = [], 0
    in_code, buf = False, []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            if in_code:
                out.append('<pre><code>' + html.escape('\n'.join(buf)) + '</code></pre>')
                buf, in_code = [], False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            buf.append(ln); i += 1; continue

        if ln.strip().startswith('|') and i+1 < len(lines) and re.match(r'^\s*\|[\s\-:|]+\|\s*$', lines[i+1]):
            head = [c.strip() for c in ln.strip().strip('|').split('|')]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            out.append('<table><thead><tr>' + ''.join('<th>%s</th>' % inline(c) for c in head)
                       + '</tr></thead><tbody>')
            for r in rows:
                out.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in r) + '</tr>')
            out.append('</tbody></table>')
            continue

        m = re.match(r'^(#{1,4})\s+(.*)$', ln)
        if m:
            lv = len(m.group(1)); out.append('<h%d>%s</h%d>' % (lv, inline(m.group(2)), lv)); i += 1; continue

        if ln.strip() == '---':
            out.append('<hr>'); i += 1; continue

        if ln.startswith('>'):
            buf2 = []
            while i < len(lines) and lines[i].startswith('>'):
                buf2.append(lines[i].lstrip('> ').rstrip()); i += 1
            out.append('<blockquote>' + '<br>'.join(inline(x) for x in buf2) + '</blockquote>')
            continue

        if re.match(r'^\s*[-*]\s+', ln):
            out.append('<ul>')
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                out.append('<li>' + inline(re.sub(r'^\s*[-*]\s+', '', lines[i])) + '</li>'); i += 1
            out.append('</ul>'); continue

        if re.match(r'^\s*\d+[.、]\s+', ln):
            out.append('<ol>')
            while i < len(lines) and re.match(r'^\s*\d+[.、]\s+', lines[i]):
                out.append('<li>' + inline(re.sub(r'^\s*\d+[.、]\s+', '', lines[i])) + '</li>'); i += 1
            out.append('</ol>'); continue

        if not ln.strip():
            i += 1; continue

        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,4}\s|[-*]\s|\d+[.、]\s|```|\||>)', lines[i]):
            para.append(lines[i].strip()); i += 1
        out.append('<p>' + '<br>'.join(inline(x) for x in para) + '</p>')
    return '\n'.join(out)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font: 15px/1.75 -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
       color: #1b1b1f; max-width: 100%; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: .5px; }
h2 { font-size: 21px; margin: 30px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #ddd; }
h3 { font-size: 17px; margin: 22px 0 8px; }
p { margin: 9px 0; }
code { background: #f2f2f5; padding: 1px 5px; border-radius: 4px; font-size: 13px;
       font-family: "SF Mono", Menlo, monospace; }
pre { background: #f7f7fa; border: 1px solid #e6e6ee; border-radius: 8px; padding: 12px 14px;
      overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 12.5px; line-height: 1.6; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13.5px; }
th, td { border: 1px solid #dcdce4; padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: #f5f5f9; font-weight: 600; }
blockquote { margin: 12px 0; padding: 10px 14px; background: #fbf8f2;
             border-left: 3px solid #e0b070; color: #333; }
hr { border: none; border-top: 1px solid #e5e5ee; margin: 26px 0; }
ul, ol { margin: 8px 0 8px 22px; padding: 0; }
li { margin: 4px 0; }
a { color: #b4600f; text-decoration: none; }
"""

def build(md_path, html_path):
    md = open(md_path, encoding='utf-8').read()
    body = convert(md)
    html_doc = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                '<title>%s</title><style>%s</style></head><body>%s</body></html>'
                % (os.path.basename(md_path), CSS, body))
    open(html_path, 'w', encoding='utf-8').write(html_doc)
    return html_path

if __name__ == '__main__':
    md, out = sys.argv[1], sys.argv[2]
    tmp = '/tmp/_md2pdf.html'
    build(md, tmp)
    chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
                    '--user-data-dir=/tmp/chpdf-profile',
                    '--no-pdf-header-footer', '--print-to-pdf=' + out,
                    'file://' + tmp], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(out, os.path.getsize(out) // 1024, 'KB')
