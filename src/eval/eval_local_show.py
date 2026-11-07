"""本地生成 + 判别器评测。

对同一批主题，用不同长度要求生成，看：
  clf     = P(毛选) 判别器分数（人写参考约 79）
  rep4    = 4-gram 重复率（人写约 0.06）
  chars   = 实际字数
  tps     = 生成速度

同时给出综合分 combined = clf × (1 - 复读惩罚)，避免「靠复读骗高分」。
"""
import io
import json
import pickle
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

CLF_PATH = r"results\mao_clf.pkl"
PROXY = "http://127.0.0.1:8081/v1/chat/completions"

TOPICS = [
    ("为人民服务", "人民、利益、牺牲、责任、同志", 300),
    ("为人民服务", "人民、利益、牺牲、责任、同志", 800),
    ("反对本本主义", "调查研究、实际、教条、结论", 300),
    ("工作方法", "群众、调查、实际、领导、问题", 300),
    ("学习和时局", "学习、分析、判断、形势", 300),
    ("批评与自我批评", "缺点、错误、改正、团结、同志", 300),
]

TEMPLATE = "以《毛泽东选集》的文风和论证方式，围绕「{topic}」写一段论述，约{n}字。可涉及：{kw}"


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


def combined(clf, rep, lo=0.09, hi=0.50):
    pen = max(0.0, min(1.0, (rep - lo) / (hi - lo)))
    return clf * (1 - pen)


def gen(prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "repeat_penalty": 1.35,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(PROXY, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    c = d["choices"][0]
    usage = d.get("usage", {})
    return c["message"]["content"], dt, usage.get("completion_tokens", 0)


def main():
    print("加载判别器 ...", flush=True)
    clf = pickle.load(open(CLF_PATH, "rb"))
    print("OK\n", flush=True)

    # 人写基线
    ref_rows = [json.loads(l) for l in
                io.open("data_mao/test.jsonl", encoding="utf-8")]
    ref_p = clf.predict_proba([clean(r["response"]) for r in ref_rows])[:, 1]
    ref_clf = float(ref_p.mean()) * 100
    ref_rep = sum(repeat_rate(r["response"]) for r in ref_rows) / len(ref_rows)
    print("=" * 78, flush=True)
    print("人写毛选参照:  clf=%.1f   repeat4=%.3f   combined=%.1f"
          % (ref_clf, ref_rep, combined(ref_clf, ref_rep)), flush=True)
    print("=" * 78, flush=True)

    rows = []
    for i, (topic, kw, n) in enumerate(TOPICS, 1):
        prompt = TEMPLATE.format(topic=topic, kw=kw, n=n)
        print("\n[%d/%d] %s (要求 %d 字)" % (i, len(TOPICS), topic, n), flush=True)
        try:
            text, dt, toks = gen(prompt, max_tokens=int(n * 1.6) + 120)
        except Exception as e:
            print("  生成失败: %s" % str(e)[:150], flush=True)
            continue

        p = float(clf.predict_proba([clean(text)])[0, 1]) * 100
        rep = repeat_rate(text)
        rec = {"topic": topic, "req": n, "chars": len(clean(text)),
               "clf": round(p, 1), "rep4": round(rep, 3),
               "combined": round(combined(p, rep), 1),
               "secs": round(dt, 1),
               "tps": round(toks / max(0.1, dt), 1), "text": text}
        rows.append(rec)
        print("  -> %d 字  clf=%.1f  rep4=%.3f  combined=%.1f  (%.1fs, %.1f t/s)"
              % (rec["chars"], rec["clf"], rec["rep4"], rec["combined"],
                 rec["secs"], rec["tps"]), flush=True)

    if not rows:
        print("没有成功的样本", flush=True)
        return

    print("\n" + "=" * 78, flush=True)
    print("%-16s %6s %7s %7s %7s %9s %8s"
          % ("主题", "要求字", "实际字", "clf", "rep4", "combined", "t/s"), flush=True)
    print("-" * 78, flush=True)
    for r in rows:
        print("%-16s %6d %7d %7.1f %7.3f %9.1f %8.1f"
              % (r["topic"], r["req"], r["chars"], r["clf"], r["rep4"],
                 r["combined"], r["tps"]), flush=True)
    print("-" * 78, flush=True)
    print("%-16s %6s %7s %7.1f %7.3f %9.1f %8s"
          % ("人写参考", "-", "-", ref_clf, ref_rep,
             combined(ref_clf, ref_rep), "-"), flush=True)

    avg_clf = sum(r["clf"] for r in rows) / len(rows)
    avg_rep = sum(r["rep4"] for r in rows) / len(rows)
    print("\n本模型均值: clf=%.1f  rep4=%.3f  combined=%.1f"
          % (avg_clf, avg_rep, combined(avg_clf, avg_rep)), flush=True)
    print("相对人写:   clf %+.1f 分（人写的 %.0f%%）"
          % (avg_clf - ref_clf, avg_clf / ref_clf * 100), flush=True)

    with io.open("results/local_eval.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "ref": {"clf": ref_clf, "rep4": ref_rep}}, f,
                  ensure_ascii=False, indent=2)
    print("\n保存 results/local_eval.json", flush=True)

    # 打印两段正文供人眼看
    print("\n" + "=" * 78, flush=True)
    for r in rows[:2]:
        print("\n【%s  要求%d字  实际%d字  clf=%.1f】"
              % (r["topic"], r["req"], r["chars"], r["clf"]), flush=True)
        print("-" * 78, flush=True)
        print(r["text"].strip()[:1200], flush=True)


if __name__ == "__main__":
    main()
