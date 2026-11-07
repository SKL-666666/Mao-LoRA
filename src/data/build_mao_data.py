#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把《毛泽东选集》四卷原文切成「主题 → 毛选体段落」训练样本。

任务形式（关键设计）：
    用户：以《毛泽东选集》的文风和论证方式，围绕「<主题>」，写一段<文体>，约 <N> 字。
    助手：<原文段落>

为什么用这个形式而不是「续写」：
  - 上一轮史记失败的根本原因是「续写」要求模型自己编史实，1B 编不出来就复读。
  - 毛选是议论文，论证结构（设问→分层→驳论→结论）可以学，
    不需要事实库支撑；主题由 prompt 给定，模型只负责「换一层毛选的皮」。
  - 这正是「模仿语言风格和部分主旨内容」的可操作化。

切分规则：
  - 按文章切；文章内按小节切；小节内在段落边界上切成 250~450 字
  - 太短的段落丢弃（<150 字），避免学成短句拼贴
  - 保留标题线索（用于提主题），但不进正文

划分：按「整篇文章」划分 train/val/test，防止同一篇文章的段落同时出现在训练和测试里。
"""
import os
import re
import sys
import json
import random
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_mao", "raw")
OUT = os.path.join(HERE, "data_mao")

VOLUMES = ["vol1", "vol2", "vol3", "vol4"]

# 文章标题行：短、不以句末标点结尾，且下一行常是（一九xx年…）
DATE_RE = re.compile(r"^[（(]\s*一九[〇○零一二三四五六七八九十百]+年")
# 小节标题：一 / 二 … 或 纯短标题
SECTION_RE = re.compile(r"^([一二三四五六七八九十]{1,3})\s+(\S.{0,30})$")
# 题解括注（"这是毛泽东为……写的"）需要剔除
NOTE_RE = re.compile(r"^[（(]\s*这是毛泽东")

# 噪声：出版社/编者加的说明
JUNK_PAT = re.compile(
    r"^(目\s*录|第[一二三四五六七八九十]+卷|出版说明|编者注|注释|"
    r"《毛泽东选集》|人民出版社|根据.{0,20}刊印)"
)


def load_volume(v):
    p = os.path.join(RAW, v + ".txt")
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def is_title(line, nxt):
    """判断是否文章标题。"""
    if not (3 <= len(line) <= 30):
        return False
    if re.search(r"[。；！？、，：”』】]$", line):
        return False
    if NOTE_RE.match(line) or JUNK_PAT.match(line):
        return False
    if DATE_RE.match(nxt or ""):
        return True
    # 有些标题下一行是空行再接日期
    return False


def split_articles(lines):
    """按卷切文章：返回 [(title, heading, [paras])]。"""
    arts = []
    cur = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        if is_title(line, nxt):
            if cur and cur["paras"]:
                arts.append(cur)
            cur = {"title": line, "date": nxt if DATE_RE.match(nxt) else "",
                   "heading": "", "paras": []}
            i += 2 if DATE_RE.match(nxt) else 1
            continue
        if cur is None:
            i += 1
            continue
        if DATE_RE.match(line) or NOTE_RE.match(line):
            i += 1
            continue
        m = SECTION_RE.match(line)
        if m and len(line) <= 34:
            cur["heading"] = line
            i += 1
            continue
        # 小节标题：短且无句末标点
        if 2 <= len(line) <= 24 and not re.search(r"[。；！？：，、”』】]$", line):
            cur["heading"] = line
            i += 1
            continue
        cur["paras"].append(line)
        i += 1
    if cur and cur["paras"]:
        arts.append(cur)
    return arts


def classify(head):
    """粗判段落文体（用于 prompt 里描述）。"""
    if re.search(r"[？?]", head[:120]) and re.search(r"(答复|回答|我们说|答案是)", head[:400]):
        return "设问式论述"
    if re.search(r"(有人说|他们以为|这种说法|是不对的|驳|错误观点|有些人说)", head[:300]):
        return "驳论"
    if re.search(r"^[（(][一二三四五六七八九十]", head) or re.search(r"第[一二三四五六七八九十]、", head[:200]):
        return "分层论证"
    if re.search(r"(总之|综上所述|结论|由此可见|因此我们可以说)", head[:400]):
        return "总结"
    if re.search(r"(必须|应当|应该|一定要|我们要)", head[:200]):
        return "方针指示"
    return "论述"


PROMPT_TEMPLATES = [
    "以《毛泽东选集》的文风和论证方式，围绕「{topic}」写一段{kind}，约 {n} 字。",
    "请用毛泽东的笔法，就「{topic}」写一段{kind}，约 {n} 字。",
    "模仿《毛泽东选集》的语言风格，围绕「{topic}」展开一段{kind}，约 {n} 字。",
    "用毛选式的说理口吻，论述「{topic}」。要求{kind}，约 {n} 字。",
    "按照《毛泽东选集》的写法，写一段关于「{topic}」的{kind}，约 {n} 字。",
    "以毛选文体写一段{kind}，主题是「{topic}」，约 {n} 字。",
]

# 「关键词展开」式 prompt：输入只给主题 + 若干关键词，输出是对应的毛选体论述。
# 这是本次任务的核心形式——内容由输入给定，模型只负责组织成毛选的语言与论证方式。
#
# 为什么不用「原文提纲」：提纲若取原文句子，等于把答案抄进 prompt，
#   模型学会的是「复制输入」，loss 会降得漂亮但什么也没学到（史记那轮的另一面教训）。
# 为什么不用「自由续写」：1B 编不出内容就复读（史记那轮已证明）。
PLAIN_TEMPLATES = [
    "以《毛泽东选集》的文风和论证方式，围绕「{topic}」写一段{kind}，约 {n} 字。可涉及：{kw}",
    "用毛选式的说理口吻谈「{topic}」，约 {n} 字，要写到这些方面：{kw}",
    "按照《毛泽东选集》的写法，就「{topic}」写一段{kind}，约 {n} 字。要点：{kw}",
    "模仿毛泽东的笔法，围绕「{topic}」展开论述，约 {n} 字，涉及：{kw}",
    "以毛选文体写一段{kind}，主题「{topic}」，约 {n} 字，内容涉及：{kw}",
]

STOPWORDS = set("""
的 了 是 在 和 与 就 也 都 而 及 或 但 很 更 最 这 那 有 无 不 为 以 于 之 其 者 被 把 从 到 对 由 因 所 使 让 给 向
我们 你们 他们 它们 自己 这个 那个 这些 那些 这样 那样 一个 一些 什么 怎么 因为 所以 但是 如果 虽然 可以 应该 必须
已经 还是 就是 不是 没有 不能 不会 这样 那样 问题 情况 方面 地方 时候 现在 过去 将来 一切 各种 许多 很多
""".split())


def make_keywords(chunk, topk=8):
    """抽取关键词。只给词，不给句子，避免答案泄漏进 prompt。"""
    import jieba
    import jieba.analyse
    try:
        kws = jieba.analyse.extract_tags(chunk, topK=topk * 2)
    except Exception:  # noqa: BLE001
        kws = []
    out = []
    for w in kws:
        if len(w) < 2 or w in STOPWORDS:
            continue
        if re.fullmatch(r"[\d\W_]+", w):
            continue
        out.append(w)
        if len(out) >= topk:
            break
    return "、".join(out)


def chunk_article(art, min_chars=220, max_chars=460):
    """在段落边界上切块。"""
    chunks = []
    buf = []
    size = 0
    for p in art["paras"]:
        if len(p) > max_chars * 1.6:
            # 超长段落自己切开
            if buf:
                chunks.append("\n".join(buf))
                buf, size = [], 0
            for k in range(0, len(p), max_chars):
                seg = p[k:k + max_chars]
                if len(seg) >= min_chars:
                    chunks.append(seg)
            continue
        if size + len(p) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p)
    if buf and size >= min_chars:
        chunks.append("\n".join(buf))
    return [c for c in chunks if len(c) >= min_chars]


def main():
    random.seed(20261106)
    all_arts = []
    for v in VOLUMES:
        lines = load_volume(v)
        arts = split_articles(lines)
        print("%s: %d 篇" % (v, len(arts)), flush=True)
        for a in arts:
            a["vol"] = v
        all_arts.extend(arts)

    # 只保留有实质正文的文章
    all_arts = [a for a in all_arts if sum(len(p) for p in a["paras"]) >= 600]
    print("有效文章: %d 篇，总字数 %d" % (
        len(all_arts), sum(sum(len(p) for p in a["paras"]) for a in all_arts)), flush=True)

    # 按整篇划分
    idx = list(range(len(all_arts)))
    random.shuffle(idx)
    n_test = max(4, len(idx) // 20)
    n_val = max(3, len(idx) // 25)
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test:n_test + n_val])

    splits = {"train": [], "val": [], "test": []}
    for i, a in enumerate(all_arts):
        split = "test" if i in test_idx else ("val" if i in val_idx else "train")
        chunks = chunk_article(a)
        for c in chunks:
            kind = classify(c)
            topic = a["title"]
            if a["heading"]:
                topic = "%s·%s" % (a["title"], a["heading"])
            kw = make_keywords(c)
            # 70% 用「关键词展开」（内容范围给定，学文风），
            # 30% 用「主题自由写」（更接近开放风格迁移，但更难）
            if kw and random.random() < 0.7:
                tpl = random.choice(PLAIN_TEMPLATES)
                prompt = tpl.format(topic=topic, kind=kind, n=len(c), kw=kw)
                form = "keywords"
            else:
                tpl = random.choice(PROMPT_TEMPLATES)
                prompt = tpl.format(topic=topic, kind=kind, n=len(c))
                form = "topic"
            splits[split].append({
                "prompt": prompt,
                "response": c,
                "meta": {"vol": a["vol"], "article": a["title"],
                         "heading": a["heading"], "kind": kind,
                         "form": form, "chars": len(c)},
            })

    os.makedirs(OUT, exist_ok=True)
    for name, rows in splits.items():
        p = os.path.join(OUT, name + ".jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        chars = sum(r["meta"]["chars"] for r in rows)
        print("%-6s %5d 条  %8d 字  -> %s" % (name, len(rows), chars, p), flush=True)

    kinds = Counter(r["meta"]["kind"] for r in splits["train"])
    print("\n文体分布:", kinds.most_common(), flush=True)

    # 抽查
    print("\n--- 抽查 3 条 ---")
    for r in splits["train"][:3]:
        print("PROMPT:", r["prompt"])
        print("RESP  :", r["response"][:160].replace("\n", " "), "...")
        print()


if __name__ == "__main__":
    main()
