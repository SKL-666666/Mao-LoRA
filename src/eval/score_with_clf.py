#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用训练好的判别器给模型输出打「像不像毛选」的分。

指标：
  clf_score   = P(毛选) * 100，主指标（判别器输出）
  repeat4     = 4-gram 重复率，防复读机
  vk_ratio    = 文言虚词密度（之/其/乃/遂/焉/矣/故/皆），辅助解释
  md_ratio    = Markdown 标记数（### / **），越少越不像模型腔
  chars       = 长度

为什么用 clf_score 作主指标：
  手写词表版把 base 判 54.69、人写判 49.63（base 更高），完全失效。
  判别器版：人写 79.8、base 16.0，差 5 倍。这才是能用的指标。

用法：
  python3 scripts/score_with_clf.py --tag ckpt_0100 --adapter outputs/lora_mao/ckpt_0100
  python3 scripts/score_with_clf.py --compare base: ckpt1:path1 ckpt2:path2
"""
import os
import re
import sys
import json
import time
import pickle
import argparse

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
CLF = os.path.join(ROOT, "results", "mao_clf.pkl")

SYSTEM = "你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。"

CLASSICAL = ["之", "其", "乃", "遂", "焉", "矣", "故", "皆", "是以", "所谓", "何以"]


def clean(t):
    return re.sub(r"\s+", "", t)


def repeat_rate(text, n=4):
    t = clean(text)
    if len(t) < n + 1:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    c = {}
    for g in grams:
        c[g] = c.get(g, 0) + 1
    dup = sum(v - 1 for v in c.values() if v > 1)
    return dup / len(grams)


def side_metrics(text):
    t = clean(text)
    n = max(1, len(t))
    vk = sum(t.count(w) for w in CLASSICAL) / n * 1000
    md = len(re.findall(r"(^#{1,4}\s|\*\*)", text, re.M))
    return {"chars": len(t), "repeat4": round(repeat_rate(text), 3),
            "vk_ratio": round(vk, 2), "md_marks": md}


def build_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(adapter):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    base = os.path.join(ROOT, "models", "MiniCPM5-1B")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False, attn_implementation="sdpa")
    if adapter and adapter not in ("base", ""):
        from peft import PeftModel
        p = adapter if os.path.isabs(adapter) else os.path.join(ROOT, adapter)
        model = PeftModel.from_pretrained(model, p)
    model.eval()
    return tok, model


def generate(tok, model, prompt_text, max_new=320):
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
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=os.path.join(ROOT, "data_mao", "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mao_clf_eval"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    clf = pickle.load(open(CLF, "rb"))
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tag = args.tag or (os.path.basename(args.adapter.rstrip("/")) or "base")
    print("[模型] %s  (%d 条)" % (args.adapter or "base", len(rows)), flush=True)
    tok, model = load_model(args.adapter)

    recs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        gen = generate(tok, model, r["prompt"], args.max_new)
        p = float(clf.predict_proba([clean(gen)])[0, 1])
        m = side_metrics(gen)
        m["clf_score"] = round(p * 100, 1)
        recs.append({"i": i, "prompt": r["prompt"], "gen": gen,
                     "ref": r["response"], "metrics": m})
        print("  [%2d/%d] %4d字 clf=%5.1f repeat=%.3f vk=%.1f md=%d (%.0fs)"
              % (i + 1, len(rows), m["chars"], m["clf_score"], m["repeat4"],
                 m["vk_ratio"], m["md_marks"], time.time() - t0), flush=True)
        if i < 2:
            print("     >> %s" % gen[:150].replace("\n", " "), flush=True)

    keys = ["clf_score", "repeat4", "vk_ratio", "md_marks", "chars"]
    agg = {k: round(sum(r["metrics"][k] for r in recs) / len(recs), 2) for k in keys}

    # 参照：人写留出集的判别器分
    ref_p = clf.predict_proba([clean(r["response"]) for r in rows])[:, 1]
    ref_agg = {"clf_score": round(float(ref_p.mean()) * 100, 1)}

    print("\n=== %s 汇总 ===" % tag, flush=True)
    for k in keys:
        print("  %-12s %8.2f" % (k, agg[k]), flush=True)
    print("  %-12s %8.1f  <-- 人写参考" % ("(ref)", ref_agg["clf_score"]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "adapter": args.adapter, "agg": agg,
                   "ref_score": ref_agg["clf_score"], "recs": recs},
                  f, ensure_ascii=False, indent=2)
    print("  保存: %s" % os.path.join(args.out, "%s.json" % tag), flush=True)


if __name__ == "__main__":
    main()
