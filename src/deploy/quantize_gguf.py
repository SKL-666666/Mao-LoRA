#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 llama-cpp-python 自带的 libllama.so 直接做 GGUF 量化（替代 llama-quantize 命令行）。

为什么要这么绕：
  云端没有编译好的 llama-quantize 可执行文件，而 git clone 源码编译会超时（130s 断连）。
  但 pip 装的 llama-cpp-python 里带着完整的 libllama.so，量化函数
  llama_model_quantize 是导出的，所以可以直接 ctypes 调用。

参数含义（对应 llama_model_quantize_params）：
  ftype : GGML_TYPE 枚举编号
          2=F16  7=Q8_0  15=Q4_K_M(=Q4_K)  14=Q4_K_S  17=Q5_K_M  12=Q4_0  3=Q4_1
  nthread : 线程数
  output_tensor_type / token_embedding_type : 0 表示跟随 ftype
"""
import argparse
import ctypes
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from llama_cpp import llama_cpp as lc

FTYPES = {
    "Q4_0": 2 + 0,      # 占位，下面用真实枚举覆盖
}

# ggml_type 真实枚举（llama.cpp gguf/constants.py GGMLQuantizationType）
GGML = {
    "F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7,
    "Q8_0": 8, "Q8_1": 9, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13,
    "Q6_K": 14, "Q8_K": 15,
}
# 注意：命令行里的 Q4_K_M 不是单一 ggml_type，而是 llama.cpp 内部的
# LLAMA_FTYPE_MOSTLY_Q4_K_M = 15。这里用 ftype 枚举，不是 ggml_type。
FTYPE = {
    "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 8, "Q5_1": 9, "Q8_0": 7,
    "Q2_K": 10, "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13,
    "Q4_K_S": 14, "Q4_K_M": 15, "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="输入 f16/f32 GGUF")
    ap.add_argument("--dst", required=True, help="输出量化 GGUF")
    ap.add_argument("--ftype", default="Q4_K_M", choices=list(FTYPE))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit("找不到输入文件: %s" % args.src)

    lib = lc._lib
    lib.llama_backend_init()

    params = lib.llama_model_quantize_default_params()
    params.ftype = FTYPE[args.ftype]
    params.nthread = args.threads
    print("量化配置: ftype=%s(%d) threads=%d" % (args.ftype, params.ftype, args.threads),
          flush=True)

    src_b = args.src.encode()
    dst_b = args.dst.encode()

    t0 = time.time()
    rc = lib.llama_model_quantize(src_b, dst_b, ctypes.byref(params))
    print("返回码: %d  耗时 %.1f 秒" % (rc, time.time() - t0), flush=True)

    if rc != 0:
        raise SystemExit("量化失败，返回码 %d" % rc)

    if not os.path.exists(args.dst):
        raise SystemExit("量化声称成功但没有输出文件")

    print("输出: %s  %.1f MB" % (args.dst, os.path.getsize(args.dst) / 1e6), flush=True)


if __name__ == "__main__":
    main()