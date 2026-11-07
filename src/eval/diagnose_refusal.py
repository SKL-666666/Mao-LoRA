import io
import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

SERVER = r"tools\llama-server.exe"
GGUF = r"outputs\gguf\mao-Q4_K_M.gguf"
PORT = 8083

SYS = "你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。"

# 复现截图里的情况，并逐项排查原因
CASES = [
    ("A 无system + 用户原话", None, "以你的口吻，写一篇“为人民服务”"),
    ("B 有system + 用户原话", SYS, "以你的口吻，写一篇“为人民服务”"),
    ("C 有system + 训练格式", SYS,
     "以《毛泽东选集》的文风和论证方式，围绕「为人民服务」写一段论述，约300字。可涉及：人民、利益、牺牲、责任、同志"),
]

proc = subprocess.Popen(
    [SERVER, "-m", GGUF, "-c", "2048", "-t", "4",
     "--host", "127.0.0.1", "--port", str(PORT), "--jinja",
     "--chat-template-kwargs", '{"enable_thinking": false}',
     "--repeat-penalty", "1.35", "--repeat-last-n", "256"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

ready = False
for _ in range(120):
    time.sleep(1)
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        continue
print("server ready:", ready, flush=True)
if not ready:
    proc.kill()
    raise SystemExit(1)


def gen(system, user, max_tokens=350):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = {"messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95,
            "repeat_penalty": 1.35, "stream": False}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    c = d["choices"][0]
    return c["message"]["content"], d.get("usage", {})


for name, sysmsg, user in CASES:
    print("=" * 72, flush=True)
    print(name, flush=True)
    print("system:", "(无)" if not sysmsg else sysmsg[:40] + "...", flush=True)
    print("-" * 72, flush=True)
    try:
        text, usage = gen(sysmsg, user)
        print(text.strip()[:700], flush=True)
        print("\n[tokens]", usage.get("completion_tokens"), flush=True)
    except Exception as e:
        print("[失败]", str(e)[:200], flush=True)
    print(flush=True)

proc.terminate()
try:
    proc.wait(timeout=10)
except Exception:
    proc.kill()
