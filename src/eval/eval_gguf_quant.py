#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Q4_K_M 量化后的精度保持情况。

两个指标（缺一不可）：
  ① perplexity：语言建模能力掉多少
     —— 但 ppl 对「风格」不敏感，模型可能仍通顺却完全不像毛选
  ② 判别器分数 clf：P(毛选) 掉多少
     —— 这才是这次任务真正关心的「精度」

只报 ppl 会得出错误结论。举个具体例子：Q4 后生成从
「现在国际上正在发生着许多新的变化」变成「当今世界形势复杂多变」，
ppl 可能只涨 0.05，但 clf 会掉 40 分。
"""
import json
import os
import pickle
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_DIR = os.path.join(ROOT, "outputs", "gguf")
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


def build_prompt(tok, text):
    # Qwen/MiniCPM 的 chat template 由 GGUF 里的 jinja 提供，
    # 这里手工拼出与训练时一致的格式
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, text))


def main():
    from llama_cpp import Llama

    clf = pickle.load(open(CLF, "rb"))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data_mao", "test.jsonl"),
                                        encoding="utf-8")][:6]
    ref_clf = float(clf.predict_proba([clean(r["response"]) for r in rows])[:, 1].mean()) * 100
    print("人写参考 clf=%.1f\n" % ref_clf, flush=True)

    targets = [("mao-f16.gguf", "f16"), ("mao-Q4_K_M.gguf", "Q4_K_M")]
    results = {}

    for fn, tag in targets:
        path = os.path.join(GGUF_DIR, fn)
        if not os.path.exists(path):
            print("[跳过] %s 不存在" % fn, flush=True)
            continue

        print("=" * 60, flush=True)
        print("[加载] %s (%.0f MB)" % (tag, os.path.getsize(path) / 1e6), flush=True)
        t0 = time.time()
        llm = Llama(model_path=path, n_ctx=1024, n_threads=8, verbose=False,
                    logits_all=True)
        print("  加载耗时 %.1f 秒" % (time.time() - t0), flush=True)

        # ---- ① 困惑度 ----
        corpus = "".join(r["response"] for r in rows)[:4000]
        try:
            ppl = llm.create_completion(prompt="", max_tokens=0)
            # llama-cpp-python 没有直接 ppl 接口，用 logits 自算
            import math
            total_logprob, total_tok = 0.0, 0
            chunk = 256
            for i in range(0, len(corpus) - chunk, chunk):
                seg = corpus[i:i + chunk]
                out = llm(seg, max_tokens=1, echo=True, logprobs=1, temperature=0)
                lp = out["choices"][0].get("logprobs", {}).get("token_logprobs") or []
                for v in lp:
                    if v is not None:
                        total_logprob += v
                        total_tok += 1
            ppl_val = math.exp(-total_logprob / total_tok) if total_tok else float("nan")
        except Exception as e:  # noqa: BLE001
            ppl_val = float("nan")
            print("  ppl 计算失败: %s" % str(e)[:100], flush=True)

        print("  困惑度 ppl = %.3f (基于 %d token)" % (ppl_val, total_tok), flush=True)

        # ---- ② 判别器分数 ----
        cls, reps, gens = [], [], []
        for i, r in enumerate(rows):
            p = build_prompt(None, r["prompt"])
            out = llm(p, max_tokens=300, temperature=0, repeat_penalty=1.15,
                      stop=["<|im_end|>", "<|im_start|>"])
            g = out["choices"][0]["text"]
            cls.append(float(clf.predict_proba([clean(g)])[0, 1]) * 100)
            reps.append(repeat_rate(g))
            gens.append(g[:250])
            print("  [%d/%d] clf=%5.1f rep=%.3f len=%d"
                  % (i + 1, len(rows), cls[-1], reps[-1], len(g)), flush=True)

        results[tag] = {
            "ppl": round(ppl_val, 3),
            "clf": round(sum(cls) / len(cls), 1),
            "repeat4": round(sum(reps) / len(reps), 3),
            "size_mb": round(os.path.getsize(path) / 1e6, 1),
            "samples": gens,
        }
        del llm
        import gc
        gc.collect()

    # ---- 对比 ----
    print("\n" + "=" * 66, flush=True)
    print("%-10s %10s %10s %10s %10s" % ("精度", "size(MB)", "ppl", "clf", "repeat4"),
          flush=True)
    print("-" * 66, flush=True)
    for tag, r in results.items():
        print("%-10s %10.1f %10.3f %10.1f %10.3f"
              % (tag, r["size_mb"], r["ppl"], r["clf"], r["repeat4"]), flush=True)
    print("%-10s %10s %10s %10.1f %10s" % ("人写", "-", "-", ref_clf, "-"), flush=True)
    print("-" * 66, flush=True)

    if "f16" in results and "Q4_K_M" in results:
        d_clf = results["Q4_K_M"]["clf"] - results["f16"]["clf"]
        d_ppl = results["Q4_K_M"]["ppl"] - results["f16"]["ppl"]
        print("\nQ4_K_M 相对 f16:", flush=True)
        print("  clf 变化    : %+.1f 分  (%.1f -> %.1f)"
              % (d_clf, results["f16"]["clf"], results["Q4_K_M"]["clf"]), flush=True)
        print("  ppl 变化    : %+.3f    (%.3f -> %.3f)"
              % (d_ppl, results["f16"]["ppl"], results["Q4_K_M"]["ppl"]), flush=True)
        print("  体积压缩    : %.1f%%  (%.0f -> %.0f MB)"
              % ((1 - results["Q4_K_M"]["size_mb"] / results["f16"]["size_mb"]) * 100,
                 results["f16"]["size_mb"], results["Q4_K_M"]["size_mb"]), flush=True)
        verdict = "保持良好" if d_clf > -5 else ("有损但可用" if d_clf > -15 else "损失较大")
        print("  结论        : %s" % verdict, flush=True)

    out_path = os.path.join(ROOT, "results", "gguf_quant_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "ref_clf": ref_clf}, f,
                  ensure_ascii=False, indent=2)
    print("\n保存: %s" % out_path, flush=True)


if __name__ == "__main__":
    main()