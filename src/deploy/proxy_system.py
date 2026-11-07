"""给 llama-server 加一层代理：自动补 system prompt，界面里什么都不用填。

为什么要这层代理：
  llama-server 的 /v1/chat/completions 只把调用方发来的 messages 交给模型。
  网页界面默认不带 system 消息，于是模型失去 LoRA 的触发条件，
  退回基座人格（问「写为人民服务」就答「我不会说这个话」）。

  这层代理在收到请求时，若 messages 里没有 system，就自动补上，
  再把请求转发给 llama-server。这样浏览器里随便怎么说都能得到毛选腔。

用法：
  1) 先起 llama-server（端口 8080）：
       .\\tools\\llama-server.exe -m outputs\\gguf\\mao-Q4_K_M.gguf --jinja ^
         --chat-template-file models\\MiniCPM5-1B\\chat_template.jinja ^
         -c 2048 -t 4 --repeat-penalty 1.35 --host 127.0.0.1 --port 8080
  2) 再跑本脚本：python proxy_system.py
  3) 浏览器打开 http://127.0.0.1:8081  （注意是 8081，不是 8080）
"""
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")

UPSTREAM = "http://127.0.0.1:8080"
LISTEN_PORT = 8081

SYSTEM = "你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # 只打一行简洁日志，避免刷屏
        print("  [proxy] " + (fmt % args), flush=True)

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        # 只对 chat/completions 补 system，其他端点原样转发
        if "/v1/chat/completions" in self.path and raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                msgs = payload.get("messages") or []
                has_system = any(m.get("role") == "system" for m in msgs)
                if not has_system:
                    payload["messages"] = [{"role": "system", "content": SYSTEM}] + msgs
                    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    print("  [proxy] injected system prompt", flush=True)
            except Exception as e:  # noqa: BLE001
                print("  [proxy] parse failed, passing through: %s" % str(e)[:80], flush=True)

        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=raw if raw else None, method=method)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                body = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            msg = ("upstream error: %s" % str(e)[:200]).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print(" Mao 代理已启动", flush=True)
    print("   上游: %s" % UPSTREAM, flush=True)
    print("   访问: http://127.0.0.1:%d" % LISTEN_PORT, flush=True)
    print("   自动补的 system: %s" % SYSTEM, flush=True)
    print("=" * 60, flush=True)
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
