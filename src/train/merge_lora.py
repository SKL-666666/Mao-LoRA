#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 LoRA 合并进基座，导出完整 HF 权重，供转 GGUF 使用。

为什么必须先合并：
  PeftModel 的 adapter 只改了 attention 的 q/k/v/o，不能直接喂给
  convert_hf_to_gguf.py（它读的是标准 HF 权重文件）。
  必须 merge_and_unload() 生成一份「含 LoRA 效果的完整权重」。

路径解析：脚本可能在项目根、src/ 或 src/<子目录>/ 下，
向上找同时含 models/ 和 src/ 的目录作为 ROOT。

输出：models/merged_mao/
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_root(start):
    d = start
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "models")) and os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = find_root(HERE)

BASE = os.path.join(ROOT, "models", "MiniCPM5-1B")
ADAPTER = os.path.join(ROOT, "adapter")
OUT = os.path.join(ROOT, "models", "merged_mao")


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("ROOT    = %s" % ROOT, flush=True)
    print("BASE    = %s" % BASE, flush=True)
    print("ADAPTER = %s" % ADAPTER, flush=True)
    print("OUT     = %s" % OUT, flush=True)

    for p, label in [(BASE, "基座"), (ADAPTER, "adapter")]:
        if not os.path.isdir(p):
            raise SystemExit("找不到%s目录: %s" % (label, p))
    if not os.path.exists(os.path.join(BASE, "model-00000-of-00001.safetensors")):
        raise SystemExit("基座权重缺失")

    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.time()

    print("\n[1/4] 加载基座 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False)

    print("[2/4] 套上 LoRA adapter ...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)

    print("[3/4] 合并权重 ...", flush=True)
    model = model.merge_and_unload()
    model.eval()
    left = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
    print("      OK: 无残留 lora 参数" if not left else
          "      [警告] 还有 %d 个 lora 参数" % len(left), flush=True)

    print("[4/4] 保存到 %s ..." % OUT, flush=True)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)

    src_ct = os.path.join(BASE, "chat_template.jinja")
    if os.path.exists(src_ct):
        import shutil
        shutil.copy(src_ct, os.path.join(OUT, "chat_template.jinja"))

    total = sum(os.path.getsize(os.path.join(OUT, f))
                for f in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, f)))
    print("\n完成，%.1f 秒，%.2f GB" % (time.time() - t0, total / 1e9), flush=True)


if __name__ == "__main__":
    main()
