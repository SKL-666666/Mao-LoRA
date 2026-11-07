"""本地完整重建：下基座 -> 合并 LoRA -> 转 GGUF -> 量化 Q4_K_M

为什么走本地：云端到本机的 tailscale 链路只有 ~40 KB/s（relay 香港），
656MB 要 4 小时以上。而基座在魔搭是 13 MB/s，adapter 只有 31.5MB（已在本地），
本地合并+量化总共约 6 分钟。
"""
import io
import os
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def find_root(start):
    """脚本在 src/<子目录>/ 下，向上找到同时含 models/ 和 src/ 的目录。"""
    d = start
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "models")) and os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


ROOT = find_root(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "MiniCPM5-1B")
REPO = "OpenBMB/MiniCPM5-1B"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}


def list_files():
    u = "https://www.modelscope.cn/api/v1/models/%s/repo/files?Revision=master" % REPO
    r = urllib.request.Request(u, headers=UA)
    j = __import__("json").loads(
        urllib.request.urlopen(r, timeout=30, context=ctx).read().decode())
    return [(f["Name"], f.get("Size", 0)) for f in j["Data"]["Files"]
            if "/" not in f["Name"]]


def download_base():
    os.makedirs(BASE, exist_ok=True)
    files = list_files()
    print("云端文件: %s" % [(n, round(s / 1048576, 1)) for n, s in files], flush=True)
    for fn, sz in files:
        if fn.startswith("."):
            continue
        dest = os.path.join(BASE, fn)
        if os.path.exists(dest) and os.path.getsize(dest) == sz and sz > 0:
            print("  cached %-45s %8.1f MB" % (fn, sz / 1048576), flush=True)
            continue
        url = ("https://www.modelscope.cn/models/%s/resolve/master/%s"
               % (REPO, urllib.parse.quote(fn)))
        t0 = time.time()
        r = urllib.request.Request(url, headers=UA)
        resp = urllib.request.urlopen(r, timeout=120, context=ctx)
        got = 0
        with open(dest + ".part", "wb") as f:
            while True:
                b = resp.read(1 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
        os.replace(dest + ".part", dest)
        dt = max(0.1, time.time() - t0)
        print("  %-45s %8.1f MB  (%.0fs, %.1f MB/s)"
              % (fn, got / 1048576, dt, got / 1048576 / dt), flush=True)


def run(cmd, cwd=None):
    print("  $ %s" % " ".join(cmd) if isinstance(cmd, list) else "  $ %s" % cmd, flush=True)
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("    [stderr] %s" % (r.stderr or "")[-800:], flush=True)
    return r.returncode


def main():
    t_all = time.time()

    print("=" * 60, flush=True)
    print("[1/4] 下载基座 MiniCPM5-1B", flush=True)
    print("=" * 60, flush=True)
    download_base()

    print("\n" + "=" * 60, flush=True)
    print("[2/4] 合并 LoRA -> merged", flush=True)
    print("=" * 60, flush=True)
    merged = os.path.join(ROOT, "models", "merged_mao")
    if os.path.exists(os.path.join(merged, "model.safetensors")):
        print("  已存在，跳过", flush=True)
    else:
        if run([sys.executable, "src/train/merge_lora.py"]) != 0:
            print("[失败] 合并", flush=True)
            return 1

    print("\n" + "=" * 60, flush=True)
    print("[3/4] 转 GGUF f16", flush=True)
    print("=" * 60, flush=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    f16 = os.path.join(ROOT, "models", "mao-f16.gguf")
    if os.path.exists(f16):
        print("  已存在，跳过", flush=True)
    else:
        env = dict(os.environ)
        env["PYTHONPATH"] = (os.path.join(ROOT, "tools", "lcpp", "gguf-py")
                             + os.pathsep + os.path.join(ROOT, "tools", "lcpp"))
        env["TOKENIZERS_PARALLELISM"] = "false"
        r = subprocess.run(
            [sys.executable, os.path.join("tools", "lcpp", "convert_hf_to_gguf.py"),
             merged, "--outfile", f16, "--outtype", "f16"],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("  [stderr] %s" % (r.stderr or "")[-1500:], flush=True)
            print("[失败] 转换", flush=True)
            return 1
        print("  OK %.1f MB" % (os.path.getsize(f16) / 1e6), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[4/4] 量化 Q4_K_M", flush=True)
    print("=" * 60, flush=True)
    q4 = os.path.join(ROOT, "models", "mao-Q4_K_M.gguf")
    if os.path.exists(q4):
        print("  已存在，跳过", flush=True)
    else:
        qz = os.path.join(ROOT, "tools", "llama-quantize.exe")
        r = subprocess.run([qz, f16, q4, "Q4_K_M", "4"], cwd=ROOT,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("  [stderr] %s" % (r.stderr or "")[-800:], flush=True)
            print("[失败] 量化", flush=True)
            return 1
        print("  OK %.1f MB" % (os.path.getsize(q4) / 1e6), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("完成，总耗时 %.1f 分钟" % ((time.time() - t_all) / 60), flush=True)
    print("=" * 60, flush=True)
    for f in ("mao-f16.gguf", "mao-Q4_K_M.gguf"):
        p = os.path.join(ROOT, "models", f)
        if os.path.exists(p):
            print("  %-20s %8.1f MB" % (f, os.path.getsize(p) / 1e6), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
