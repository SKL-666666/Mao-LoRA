#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""毛选风格 LoRA 微调训练器。

沿用前几轮踩坑后的固定配置，不重复试错：
  ① lr=2e-5（5e-5 会震荡）
  ② 只训 attention（q/k/v/o），不碰 MLP（改 MLP 会伤能力）
  ③ 每 20 步存 ckpt，每 25 步存一次 val_loss + 生成样例
  ④ val_loss 只作参考，选 checkpoint 靠人眼看生成

与史记那轮的区别（这是这次能不能成的关键）：
  - 数据 1645 条 / 67 万字（史记只有 351 条 / 8.7 万字）
  - 任务形式是「主题+关键词 → 毛选体论述」，不是开放续写
    （开放续写要求模型自己编史实，1B 编不出来就复读）

数据格式 data_mao/{train,val}.jsonl：
  {"prompt": "...", "response": "...", "meta": {...}}

用法：
  python train_mao.py --epochs 3 --lr 2e-5
"""
import os
import sys
import json
import time
import math
import random
import argparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE

SYSTEM = "你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=os.path.join(ROOT, "models", "MiniCPM5-1B"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data_mao"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "lora_mao"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, str(args.cpu_threads))
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    t0 = time.time()
    print("[env] threads=%d device=cpu" % args.cpu_threads, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def build_prompt(prompt_text):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    def prep(path, limit=0):
        out = []
        if not os.path.exists(path):
            return out
        skipped = 0
        for line in open(path, encoding="utf-8"):
            if limit and len(out) >= limit:
                break
            r = json.loads(line)
            p_txt = build_prompt(r["prompt"])
            full = p_txt + r["response"] + (tok.eos_token or "")
            fid = tok(full, add_special_tokens=False)["input_ids"][: args.max_len]
            pid = tok(p_txt, add_special_tokens=False)["input_ids"]
            p = min(len(pid), len(fid))
            if fid[:p] != pid:
                p = 0
                for a, b in zip(fid, pid):
                    if a != b:
                        break
                    p += 1
            if len(fid) - p < 30:
                skipped += 1
                continue
            out.append({"input_ids": fid, "labels": [-100] * p + fid[p:]})
        if skipped:
            print("[prep] %s 跳过 %d 条（响应被截断）" % (os.path.basename(path), skipped),
                  flush=True)
        return out

    print("[prep] tokenizing ...", flush=True)
    train_data = prep(os.path.join(args.data_dir, "train.jsonl"))
    val_data = prep(os.path.join(args.data_dir, "val.jsonl"), args.n_val)
    print("[prep] train=%d val=%d  (%.0fs)"
          % (len(train_data), len(val_data), time.time() - t0), flush=True)
    if not train_data:
        raise SystemExit("没有训练数据")

    class DS(Dataset):
        def __init__(self, d):
            self.d = d

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            return self.d[i]

    def collate(batch):
        ml = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            pad = ml - len(b["input_ids"])
            ii.append(b["input_ids"] + [tok.pad_token_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            am.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}

    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(DS(train_data), batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, generator=g)
    dl_val = DataLoader(DS(val_data), batch_size=1, shuffle=False, collate_fn=collate)

    print("[model] loading ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False, attn_implementation="sdpa")
    model.config.use_cache = False
    model.to("cpu")

    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print("[model] 从 %s 继续训练" % args.resume, flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.alpha,
            lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(dl) / args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    lr_min = args.lr * 0.1

    def set_lr(s):
        frac = min(1.0, s / max(1, total_steps))
        lr = lr_min + 0.5 * (args.lr - lr_min) * (1 + math.cos(math.pi * frac))
        for grp in opt.param_groups:
            grp["lr"] = lr
        return lr

    os.makedirs(args.out, exist_ok=True)

    def save(tag, step, ep):
        p = os.path.join(args.out, tag)
        os.makedirs(p, exist_ok=True)
        model.save_pretrained(p)
        tok.save_pretrained(p)
        json.dump({"step": step, "epoch": ep, "args": vars(args)},
                  open(os.path.join(p, "trainer_state.json"), "w"), ensure_ascii=False)

    # 抽检 prompt 固定不变，便于跨 checkpoint 横向比较
    SAMPLE_PROMPT = ("以《毛泽东选集》的文风和论证方式，围绕「工作方法」写一段论述，"
                     "约 300 字。可涉及：群众、调查、实际、领导、问题")
    SAMPLE_PROMPT2 = "按照《毛泽东选集》的写法，写一段关于「学习和时局」的论述，约 300 字。"

    def sample(text_prompt, max_new=220):
        model.eval()
        p = build_prompt(text_prompt)
        ids = tok(p, add_special_tokens=False, return_tensors="pt",
                  return_token_type_ids=False)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.15,
                                 pad_token_id=tok.eos_token_id or 0)
        model.train()
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in dl_val:
                tot += float(model(**b).loss)
                n += 1
        model.train()
        return tot / max(1, n)

    print("[train] %d 样本 x %.1f epoch = %d 步"
          % (len(dl), args.epochs, total_steps), flush=True)
    print("[base] val_loss=%.4f" % eval_loss(), flush=True)

    step, run_loss, accum = 0, 0.0, 0
    tt = time.time()
    stop = False
    for ep in range(int(math.ceil(args.epochs))):
        for b in dl:
            loss = model(**b).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                accum = 0
                continue
            (loss / args.grad_accum).backward()
            run_loss += float(loss)
            accum += 1
            if accum >= args.grad_accum:
                lr_now = set_lr(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                print("  [step %d/%d] loss=%.4f lr=%.2e (%.0fs)"
                      % (step, total_steps, run_loss / args.grad_accum, lr_now,
                         time.time() - tt), flush=True)
                run_loss = 0.0

                if step % args.save_every == 0:
                    save("ckpt_%04d" % step, step, ep)
                    save("last", step, ep)
                    print("    [ckpt] step %d" % step, flush=True)

                if step % args.sample_every == 0:
                    vl = eval_loss()
                    s1 = sample(SAMPLE_PROMPT)
                    print("    [eval] val_loss=%.4f" % vl, flush=True)
                    print("    [样例1] %s" % s1[:220].replace("\n", " "), flush=True)
                    if step % (args.sample_every * 2) == 0:
                        s2 = sample(SAMPLE_PROMPT2)
                        print("    [样例2] %s" % s2[:220].replace("\n", " "), flush=True)

                if step >= total_steps:
                    stop = True
                    break
        if stop:
            break

    save("final", step, args.epochs)
    print("[done] %d 步，耗时 %.0f 分钟，输出 %s"
          % (step, (time.time() - tt) / 60, args.out), flush=True)


if __name__ == "__main__":
    main()
