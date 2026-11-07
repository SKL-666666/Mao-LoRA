"""启动 server + proxy，跑完评测后自动清理。不依赖外部 bat。"""
import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

SRV = r"tools\llama-server.exe"
GGUF = r"outputs\gguf\mao-Q4_K_M.gguf"
TPL = r"models\MiniCPM5-1B\chat_template.jinja"


def wait_health(port, tries=180):
    for _ in range(tries):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def main():
    # 清掉可能残留的进程
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)

    print("[1/3] 启动 llama-server :8080 ...", flush=True)
    srv = subprocess.Popen(
        [SRV, "-m", GGUF, "--jinja", "--chat-template-file", TPL,
         "-c", "2048", "-t", "4", "--repeat-penalty", "1.35",
         "--repeat-last-n", "256", "--top-k", "40", "--top-p", "0.95",
         "--min-p", "0.05", "--host", "127.0.0.1", "--port", "8080"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = wait_health(8080)
    print("    ready:", ok, flush=True)
    if not ok:
        srv.kill()
        return 1

    print("[2/3] 启动代理 :8081 ...", flush=True)
    px = subprocess.Popen([sys.executable, "proxy_system.py"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    print("[3/3] 开始评测\n", flush=True)
    try:
        subprocess.run([sys.executable, "eval_local_show.py"], check=False)
    finally:
        for p in (px, srv):
            try:
                p.terminate()
                p.wait(timeout=8)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
