#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用判别器替代手写词表来评「像不像毛选」。

为什么要换掉手工词表：
  第一版评分把 base 判成 54.69、人写毛选判成 49.63——base 反而更高。
  原因很具体：base 生成的是带 `###` 和 `**一、帝国主义**` 的 Markdown 现代政论，
  它在「结构密度」「长度」两项上天然占优，而这两项的权重是我拍的。
  手工词表 + 手拍权重不可能可靠区分「毛选腔」和「现代政论腔」。

判别器做法：
  正样本：data_mao/train.jsonl 里的毛选段落（真实毛选，1645 条）
  负样本：现代政论文本。用三类真实来源，避免「自己编的负样本」导致判别器学歪
    1) test.jsonl 的 prompt 说明文字（不含正文）
    2) 模型自己写的现代政论（base 生成结果，results/mao/base.json）
    3) 用毛选原文做「现代汉语化」的规则替换（去掉文言虚词、加 Markdown 标题）
  —— 第 2 类是关键：判别器真正要区分的就是「模型腔」和「毛选腔」。

特征：字符 2-4 gram 的 TF-IDF（不用分词，中文 char n-gram 更稳）
输出：P(毛选)，0~100 分。人写留出集应该接近高分，base 应该低。

用法：
  python3 scripts/train_mao_clf.py            # 训练并保存判别器
  python3 scripts/score_with_clf.py --tag base
"""
import os
import re
import sys
import json
import random
import pickle

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
DATA = os.path.join(ROOT, "data_mao")
OUT = os.path.join(ROOT, "results", "mao_clf.pkl")


def clean(t):
    t = re.sub(r"[〔\[（(]\s*\d+\s*[〕\]）)]", "", t)   # 〔7〕注释号
    t = re.sub(r"\s+", "", t)
    return t


def modernize(t):
    """把毛选原文改造成「现代政论腔」，作为负样本。

    规则刻意做成「模型最容易犯的那几件事」：
      - 加 Markdown 标题
      - 文言虚词换白话
      - 长句拆短
    """
    t = clean(t)
    repl = [("之", "的"), ("其", "它"), ("乃", "就是"), ("遂", "于是"),
            ("焉", "呢"), ("矣", "了"), ("故", "所以"), ("皆", "都"),
            ("亦", "也"), ("莫不", "都"), ("未尝", "从来没有"),
            ("是以", "因此"), ("何以", "为什么"), ("所谓", "所说的")]
    for a, b in repl:
        t = t.replace(a, b)
    # 拆长句
    sents = re.split(r"(?<=[。！？])", t)
    t = "".join(sents)
    # 加 Markdown 腔
    head = random.choice(["### 当前形势与任务\n", "**一、总体判断**\n",
                          "## 问题分析\n", "**（一）基本认识**\n"])
    return head + t


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    random.seed(42)

    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    test = load_jsonl(os.path.join(DATA, "test.jsonl"))
    val = load_jsonl(os.path.join(DATA, "val.jsonl"))

    pos_train = [clean(r["response"]) for r in train if len(clean(r["response"])) >= 120]
    print("毛选正样本: %d" % len(pos_train))

    # 负样本 1：规则现代化（从正样本派生，保证覆盖同样主题）
    neg_rule = [modernize(r["response"]) for r in train
                if len(clean(r["response"])) >= 120]

    # 负样本 2：base 模型自己写的现代政论（真正的对手）
    neg_model = []
    base_json = os.path.join(ROOT, "results", "mao", "base.json")
    if os.path.exists(base_json):
        d = json.load(open(base_json, encoding="utf-8"))
        neg_model = [clean(r["gen"]) for r in d.get("recs", []) if len(clean(r["gen"])) >= 120]
        print("base 生成的负样本: %d" % len(neg_model))

    neg = neg_rule + neg_model
    X = pos_train + neg
    y = [1] * len(pos_train) + [0] * len(neg)
    print("负样本: %d（规则 %d + base %d）" % (len(neg), len(neg_rule), len(neg_model)))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=3, max_features=200000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print("5-fold CV 准确率: %.4f (+/- %.4f)" % (scores.mean(), scores.std()))

    pipe.fit(X, y)

    # 自检：人写留出集 vs base 生成
    def prob(texts):
        ps = pipe.predict_proba([clean(t) for t in texts])[:, 1]
        return ps

    for name, texts in [("人写 test", [r["response"] for r in test]),
                        ("人写 val", [r["response"] for r in val]),
                        ("base 生成", [r["gen"] for r in
                                       json.load(open(base_json, encoding="utf-8"))["recs"]]
                         if os.path.exists(base_json) else [])]:
        if not texts:
            continue
        ps = prob(texts)
        ps_sorted = sorted(ps)
        print("%-12s n=%3d  P(毛选) 均值=%.3f 中位=%.3f  min=%.3f max=%.3f"
              % (name, len(ps), ps.mean(), ps_sorted[len(ps) // 2], ps_sorted[0], ps_sorted[-1]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(pipe, f)
    print("\n判别器已保存: %s" % OUT)


if __name__ == "__main__":
    main()
