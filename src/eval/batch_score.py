#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量评测多个 checkpoint，输出「clf 分数 vs 复读率」曲线，找最佳停止点。

为什么需要综合分：
  ckpt_0400 的 clf=68.3 看着不错，但 repeat4=0.17（人写 p90 只有 0.09）。
  抽查发现样本 1 的 clf 高到 73.3，而 repeat4=0.487——近一半 4-gram 在重复，
  它靠「像毛选的话反复说」骗到了高分。判别器只判断文体，不知道文本在原地打转。

  combined = clf_score * (1 - penalty)
  penalty  = clip((repeat4 - 0.09) / (0.50 - 0.09), 0, 1)
  → repeat4 <= 0.09（人写正常范围）不罚；到 0.50 罚满，综合分归零。

用法：
  python3 scripts/batch_score.py --ckpts 0200 0400 0600 0800 final --n 8
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


def combined(clf_score, rep, lo=0.09, hi=0.50):
    pen = max(0.0, min(1.0, (rep - lo) / (hi - lo)))
    return round(clf_score * (1 - pen), 1)


def build_prompt(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8, help="每个 checkpoint 测几条")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mao_curve.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(args.threads)

    clf = pickle.load(open(CLF, "rb"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data_mao", "test.jsonl"),
                                        encoding="utf-8")][: args.n]

    base = os.path.join(ROOT, "models", "MiniCPM5-1B")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 参照值：人写在同样 n 条上的表现
    ref_clf = float(clf.predict_proba([clean(r["response"]) for r in rows])[:, 1].mean()) * 100
    ref_rep = sum(repeat_rate(r["response"]) for r in rows) / len(rows)
    print("人写参考: clf=%.1f repeat4=%.3f combined=%.1f"
          % (ref_clf, ref_rep, combined(ref_clf, ref_rep)), flush=True)

    results = []
    for ck in args.ckpts:
        path = os.path.join(ROOT, "outputs", "lora_mao", ck) if ck != "base" else None
        if path and not os.path.isdir(path):
            print("[跳过] %s 不存在" % ck, flush=True)
            continue

        print("\n[加载] %s" % ck, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            trust_remote_code=False, attn_implementation="sdpa")
        if path:
            model = PeftModel.from_pretrained(model, path)
        model.eval()

        cls, reps, gens = [], [], []
        t0 = time.time()
        for i, r in enumerate(rows):
            p = build_prompt(tok, r["prompt"])
            ids = tok(p, add_special_tokens=False, return_tensors="pt",
                      return_token_type_ids=False)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new,
                                     do_sample=False, repetition_penalty=1.15,
                                     pad_token_id=tok.eos_token_id or 0)
            g = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            cls.append(float(clf.predict_proba([clean(g)])[0, 1]) * 100)
            reps.append(repeat_rate(g))
            gens.append(g[:200])
            print("  [%2d/%d] clf=%5.1f rep=%.3f (%.0fs)"
                  % (i + 1, len(rows), cls[-1], reps[-1], time.time() - t0), flush=True)

        m_clf = sum(cls) / len(cls)
        m_rep = sum(reps) / len(reps)
        rec = {"ckpt": ck, "clf": round(m_clf, 1), "repeat4": round(m_rep, 3),
               "combined": combined(m_clf, m_rep), "samples": gens}
        results.append(rec)
        print("  => %s: clf=%.1f rep=%.3f combined=%.1f"
              % (ck, rec["clf"], rec["repeat4"], rec["combined"]), flush=True)

        del model
        import gc
        gc.collect()

    results.append({"ckpt": "人写", "clf": round(ref_clf, 1),
                    "repeat4": round(ref_rep, 3),
                    "combined": combined(ref_clf, ref_rep), "samples": []})

    print("\n" + "=" * 66, flush=True)
    print("%-10s %8s %10s %10s" % ("ckpt", "clf", "repeat4", "combined"), flush=True)
    print("-" * 66, flush=True)
    for r in results:
        print("%-10s %8.1f %10.3f %10.1f"
              % (r["ckpt"], r["clf"], r["repeat4"], r["combined"]), flush=True)
    best = max([r for r in results if r["ckpt"] != "人写"], key=lambda x: x["combined"])
    print("-" * 66, flush=True)
    print("最佳: %s  combined=%.1f" % (best["ckpt"], best["combined"]), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best["ckpt"],
                   "ref": {"clf": ref_clf, "repeat4": ref_rep}}, f,
                  ensure_ascii=False, indent=2)
    print("保存: %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
