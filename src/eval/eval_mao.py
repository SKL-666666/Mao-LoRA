#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""毛选风格迁移评测：可量化指标 + 并排生成对比。

为什么不用「人工判分」或「LLM 裁判」：
  前几轮吃过两次亏——regex 判分双向出错；LLM 裁判分数与 regex 差 15 分。
  风格迁移不考核「答对与否」，只考核「像不像」，这件事本身是统计特征，
  所以直接用可复算的文本统计量，不存在判分争议。

指标（都在 held-out test.jsonl 上算）：
  1) 毛选特征词密度   —— 但/所以/因此/必须/应当/一切/群众/斗争/问题/矛盾…
  2) 文言/书面腔密度   —— 之/其/乃/遂/焉/矣/者也/不可/未/所
  3) 白话口语词密度   —— 的/了/是/我们/这个/什么/吧/呢/啊（越低越好）
  4) 结构化标记密度   —— 第一、第二、其三、（一）（二）、总之、综上所述
  5) 设问密度         —— "？" 出现频率（毛选极爱用设问）
  6) 句长分布         —— 平均句长、短句（<15字）占比
  7) 复读率           —— 4-gram 重复率（判断有没有退化成复读机）
  8) 与训练语料的 n-gram 重合度 —— 判断是「学到风格」还是「背原文」

用法：
  python eval_mao.py --model base --tag base
  python eval_mao.py --model outputs/lora_mao/ckpt_0100 --tag ckpt100
  python eval_mao.py --compare base ckpt100
"""
import os
import re
import sys
import json
import time
import argparse
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE

SYSTEM = "你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。"

# ---- 词表 ----
MAO_WORDS = ["但是", "所以", "因此", "必须", "应当", "应该", "一切", "群众", "斗争",
             "问题", "矛盾", "革命", "工作", "同志", "不能", "不是", "就是",
             "只有", "才能", "如果", "既然", "否则", "而且", "并且", "然而", "可是",
             "这就是说", "换句话说", "总而言之"]
# 文言腔只用多字组合。
# 单字（之/其/故/皆/乃）会把「其实」「尤其」「之后」「故事」全算进去，
# 早先版本因此把一段纯口语判成 classical_density=33.9，必须避免。
CLASSICAL = ["者也", "不可", "未有", "所谓", "何以", "是以", "然后", "于是",
             "莫不", "未尝", "矣", "焉", "乎", "矣乎", "之谓", "若夫", "盖",
             "皆", "遂", "乃", "故曰", "是以", "所谓"]
# 注意：不能把「的/了/是/我们/这个」算进口语——毛选里这些同样是高频词。
COLLOQUIAL = ["吧", "呢", "啊", "啦", "嘛", "挺", "其实", "就是说", "然后呢",
              "哈", "嘿", "哦", "呀", "呃", "哎", "挺好", "特别好", "真的是"]
STRUCT = ["第一", "第二", "第三", "其一", "其二", "（一）", "（二）", "（三）",
          "总之", "综上所述", "由此看来", "一方面", "另一方面", "首先", "其次", "最后"]


def density(text, words):
    n = len(text)
    if n == 0:
        return 0.0
    c = sum(text.count(w) for w in words)
    return c / n * 1000.0   # 每千字


def split_sents(text):
    return [s for s in re.split(r"[。！？；\n]", text) if s.strip()]


def repeat_rate(text, n=4):
    """4-gram 重复率：重复出现的 n-gram 占全部 n-gram 的比例。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < n + 1:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    dup = sum(v - 1 for v in c.values() if v > 1)
    return dup / len(grams)


def ngram_overlap(gen, refs, n=8):
    """生成文本的 n-gram 有多少比例出现在参考语料里（背原文检测）。"""
    t = re.sub(r"\s+", "", gen)
    if len(t) < n:
        return 0.0
    grams = set(t[i:i + n] for i in range(0, len(t) - n + 1, 3))
    if not grams:
        return 0.0
    hit = sum(1 for g in grams if any(g in r for r in refs))
    return hit / len(grams)


def metrics(text, refs):
    sents = split_sents(text)
    lens = [len(s) for s in sents] or [0]
    return {
        "chars": len(text),
        "mao_density": round(density(text, MAO_WORDS), 2),
        "classical_density": round(density(text, CLASSICAL), 2),
        "colloquial_density": round(density(text, COLLOQUIAL), 2),
        "struct_density": round(density(text, STRUCT), 2),
        "q_marks_per100": round(text.count("？") / max(1, len(text)) * 100, 2),
        "avg_sent_len": round(sum(lens) / len(lens), 1),
        "short_sent_ratio": round(sum(1 for l in lens if l < 15) / len(lens), 2),
        "repeat4": round(repeat_rate(text, 4), 3),
        "ngram8_overlap": round(ngram_overlap(text, refs, 8), 3),
        "mao_score": 0.0,   # 下面算
    }


def mao_score(m):
    """综合「像毛选」的程度，0~100。

    标定：以人写的毛选（data_mao/test.jsonl 的 37 条 response）的指标分布为基准，
    把每个指标线性映射到 [0,1] 再加权。
    这样「人写的毛选」中位数 ≈ 55~60 分，是唯一可解释的参照点。

    ⚠️ 早先用了拍脑袋的固定阈值，导致人写毛选只得 33 分、和口语文本拉不开差距，
    已改为按人写分布标定。
    """
    def norm(x, lo, hi):
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    s = 0.0
    s += norm(m["mao_density"], 8.6, 48.1) * 22       # 毛选特征词
    s += norm(m["struct_density"], 0.0, 6.88) * 13    # 结构化标记
    s += norm(m["q_marks_per100"], 0.0, 0.87) * 8     # 设问
    s += norm(m["classical_density"], 1.0, 15.0) * 15 # 文言腔
    s += max(0.0, 1 - norm(m["colloquial_density"], 0.0, 6.0)) * 12  # 口语词（越低越好）
    s += max(0.0, 1 - norm(m["repeat4"], 0.05, 0.12)) * 20           # 复读（严厉）
    s += norm(m["chars"], 150, 413) * 10                             # 长度

    # 长度折扣：密度类指标在短文本上会被放大（50 字里出现 3 个结构词 = 60/千字），
    # 不折扣会让「一句话白话」拿到中等分数。人写样本中位数 413 字，按 250 字封顶折算。
    s *= min(1.0, m["chars"] / 250.0)

    if m["chars"] < 80:
        s *= 0.5
    return round(s, 1)


def build_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def load_model(model_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    base = os.path.join(ROOT, "models", "MiniCPM5-1B")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False, attn_implementation="sdpa")
    if model_path != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return tok, model


def generate(tok, model, prompt_text, max_new=300):
    import torch
    p = build_prompt(tok, prompt_text)
    ids = tok(p, add_special_tokens=False, return_tensors="pt",
              return_token_type_ids=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.15,
                             pad_token_id=tok.eos_token_id or 0)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=os.path.join(ROOT, "data_mao", "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mao"))
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    # 参考语料：训练集全部 response（用于 n-gram 重合检测）
    refs = []
    tr = os.path.join(ROOT, "data_mao", "train.jsonl")
    if os.path.exists(tr):
        refs = [json.loads(l)["response"] for l in open(tr, encoding="utf-8")]

    targets = args.compare if args.compare else [args.model]
    all_res = {}

    for target in targets:
        tag = args.tag or os.path.basename(target.rstrip("/")) or target
        path = os.path.join(ROOT, "models", "MiniCPM5-1B") if target == "base" else \
            (target if os.path.isabs(target) else os.path.join(ROOT, target))
        print("\n" + "=" * 60, flush=True)
        print("[模型] %s  (tag=%s)" % (target, tag), flush=True)
        tok, model = load_model(target)

        recs = []
        t0 = time.time()
        for i, r in enumerate(rows):
            gen = generate(tok, model, r["prompt"], args.max_new)
            m = metrics(gen, refs)
            m["mao_score"] = mao_score(m)
            recs.append({"i": i, "prompt": r["prompt"], "gen": gen,
                         "ref": r["response"], "metrics": m})
            print("  [%2d/%d] %4d字 score=%5.1f repeat=%.3f (%.0fs)"
                  % (i + 1, len(rows), m["chars"], m["mao_score"], m["repeat4"],
                     time.time() - t0), flush=True)
            if i == 0:
                print("     样例: %s" % gen[:160].replace("\n", " "), flush=True)

        # 汇总
        keys = ["mao_density", "classical_density", "colloquial_density",
                "struct_density", "q_marks_per100", "avg_sent_len",
                "short_sent_ratio", "repeat4", "ngram8_overlap", "mao_score", "chars"]
        agg = {k: round(sum(r["metrics"][k] for r in recs) / len(recs), 2) for k in keys}

        # 与测试集参考文本自身的指标做对照（"人写的毛选"应该长什么样）
        ref_metrics = [metrics(r["response"], refs) for r in rows]
        ref_agg = {k: round(sum(m[k] for m in ref_metrics) / len(ref_metrics), 2)
                   for k in keys}
        ref_agg["mao_score"] = round(
            sum(mao_score(metrics(r["response"], refs)) for r in rows) / len(rows), 2)

        all_res[tag] = {"agg": agg, "ref_agg": ref_agg, "recs": recs}
        print("\n  --- %s 均值 ---" % tag, flush=True)
        for k in keys:
            print("    %-20s %8.2f   (人写参考: %8.2f)" % (k, agg[k], ref_agg[k]),
                  flush=True)

        with open(os.path.join(args.out, "%s.json" % tag), "w", encoding="utf-8") as f:
            json.dump({"agg": agg, "ref_agg": ref_agg, "recs": recs}, f,
                      ensure_ascii=False, indent=2)

    if len(all_res) > 1:
        print("\n" + "=" * 60, flush=True)
        print("对比汇总", flush=True)
        keys = ["mao_score", "mao_density", "struct_density", "q_marks_per100",
                "colloquial_density", "repeat4", "ngram8_overlap", "chars"]
        header = "%-20s" % "指标" + "".join("%12s" % t for t in all_res) + "%12s" % "人写参考"
        print(header, flush=True)
        first = list(all_res.values())[0]
        for k in keys:
            line = "%-20s" % k
            for t, d in all_res.items():
                line += "%12.2f" % d["agg"][k]
            line += "%12.2f" % first["ref_agg"][k]
            print(line, flush=True)


if __name__ == "__main__":
    main()
