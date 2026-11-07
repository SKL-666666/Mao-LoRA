#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取《毛泽东选集》四卷全文（求是网官方源）。

页面结构：
  https://www.qstheory.cn/books/2019-07/31/c_<ID>.htm      第1页
  https://www.qstheory.cn/books/2019-07/31/c_<ID>_N.htm   第N页（N>=2）
正文段落全部用 <p> 包裹；每页结尾自带「注释」区（〔1〕...）、题解括号段、分页导航。

输出：
  data_mao/raw/vol1.txt .. vol4.txt   每卷纯文本（保留段落换行）
  data_mao/raw/meta.json              {卷: {id, pages, paras, chars}}
"""
import os
import re
import sys
import json
import time
import ssl
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_mao", "raw")

VOLUMES = {
    "vol1": "1119448591",
    "vol2": "1119478087",
    "vol3": "1119479227",
    "vol4": "1119479873",
}
BASE = "https://www.qstheory.cn/books/2019-07/31/c_%s%s.htm"  # (id, "" or "_N")


def get(url, timeout=25, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=timeout, context=ctx).read()
            for enc in ("utf-8", "gb18030"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 + i * 1.5)
    raise last


def page_url(cid, n):
    return BASE % (cid, "" if n == 1 else "_%d" % n)


def html_unescape(s):
    for a, b in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&ldquo;", "“"), ("&rdquo;", "”"),
                 ("&mdash;", "—"), ("&hellip;", "…")]:
        s = s.replace(a, b)
    return s


def extract_title(html):
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return html_unescape(m.group(1).strip()) if m else ""


TAG_RE = re.compile(r"<(p|strong|b|h1|h2|h3)[^>]*>(.*?)</\1>", re.S | re.I)


def extract_body(html):
    """按文档顺序抽取正文块。

    求是网没有稳定的正文容器（class 是 Bootstrap 的 row/container），
    并且章标题用 <strong> 而不是 <p>（例如「论持久战」被 <strong> 包着），
    只抓 <p> 会漏掉标题。因此按 <p>/<strong>/<b>/<h1..3> 的文档顺序统一抽取。

    注意：早先版本用 `<div class="content...">(.*?)</div>` 做容器截断，
    非贪婪匹配会在第一个 </div> 处停下，导致正文被截成几百字——已废弃。
    """
    out = []
    seen_pos = -1
    for m in TAG_RE.finditer(html):
        t = m.group(2)
        t = re.sub(r"<script.*?</script>", "", t, flags=re.S | re.I)
        t = re.sub(r"<[^>]+>", "", t)
        t = html_unescape(t)
        t = t.replace("\u3000", " ").replace("\xa0", " ")
        t = re.sub(r"[ \t]+", " ", t).strip()
        if t:
            out.append(t)
        seen_pos = m.end()

    if not out:
        for p in re.split(r"<br\s*/?>", html):
            t = re.sub(r"<[^>]+>", "", p)
            t = html_unescape(t).strip()
            if t:
                out.append(t)
    return out


NOISE_PAT = re.compile(
    r"^(来源：|作者：|【网站声明】|【纠错】|【打印】|上一页|下一页|返回|"
    r"\d{4}年\d{2}月\d{2}日\s|分享到|扫一扫|责任编辑|原标题)"
)


def is_noise(t):
    if re.fullmatch(r"《毛泽东选集第[一二三四]卷》", t):
        return True
    if NOISE_PAT.match(t):
        return True
    if re.fullmatch(r"[\d\s]+", t):          # 纯页码
        return True
    if len(t) <= 1:
        return True
    return False


def clean_page(paras):
    """单页清洗：去导航噪声 → 截断本页注释区 → 去掉孤立注释条目。

    必须逐页做：每页结尾都自带「注释」区，若在整卷上截断，
    第 2 页遇到「注释」就会把第 3~28 页全部丢掉（已踩过这个坑）。
    """
    keep = [t for t in paras if not is_noise(t)]

    cut = len(keep)
    for i, t in enumerate(keep):
        if t.strip() == "注释":
            cut = i
            break
    keep = keep[:cut]

    keep = [t for t in keep if not re.match(r"^[〔\[（(]\s*\d+\s*[〕\]）)]", t)]
    return keep


def fetch_volume(name, cid):
    print("[%s] id=%s 探测页数..." % (name, cid), flush=True)
    expect = "毛泽东选集"
    n = 1
    pages = []
    while True:
        url = page_url(cid, n)
        try:
            h = get(url, timeout=20)
        except Exception as e:  # noqa: BLE001
            print("   第 %d 页失败（当作结束）: %s" % (n, str(e)[:60]), flush=True)
            break
        title = extract_title(h)
        if expect not in title:
            print("   第 %d 页标题不符（结束）: %s" % (n, title[:40]), flush=True)
            break
        pages.append(h)
        n += 1
        if n > 200:
            break
        time.sleep(0.25)

    print("   共 %d 页" % len(pages), flush=True)

    all_paras = []
    for h in pages:
        all_paras.extend(clean_page(extract_body(h)))

    text = "\n".join(all_paras)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return {"id": cid, "pages": len(pages), "paras": len(all_paras), "chars": len(text)}


def main():
    meta = {}
    for name, cid in VOLUMES.items():
        try:
            meta[name] = fetch_volume(name, cid)
            print("   -> %d 段 / %d 字" % (meta[name]["paras"], meta[name]["chars"]), flush=True)
        except Exception as e:  # noqa: BLE001
            print("[%s] 失败: %s" % (name, str(e)[:120]), flush=True)
            meta[name] = {"id": cid, "error": str(e)[:200]}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total = sum(v.get("chars", 0) for v in meta.values())
    print("\n总计 %d 字" % total, flush=True)
    for k, v in meta.items():
        print("  %s: %s" % (k, v), flush=True)


if __name__ == "__main__":
    main()
