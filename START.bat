@echo off
REM ============================================================
REM  Mao LoRA - one click start
REM ============================================================
REM  This folder is self-contained. Just run this file.
REM
REM  Pure ASCII on purpose: cmd.exe reads .bat as ANSI/GBK and
REM  UTF-8 Chinese would break parsing.
REM
REM  Two processes:
REM    8080 = llama-server (model)
REM    8081 = proxy that auto-injects the system prompt
REM  Open 8081 in the browser.
REM
REM  Required flags and why:
REM    --chat-template-file : patched template that permanently disables
REM        thinking. Without it the web UI does not send enable_thinking
REM        and the model burns all tokens on
REM        "Okay, let me try to think about this..." in English.
REM    --repeat-penalty 1.35 : training used 1.15, but Q4 quantisation
REM        makes the model repeat itself; 1.35 holds it back.
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   Mao LoRA  Q4_K_M  656 MB
echo ============================================================
echo   Browser:  http://127.0.0.1:8081
echo   Stop:     close the two minimised windows
echo.
echo   System prompt is injected automatically.
echo   Leave the System Prompt box blank in the web UI.
echo ============================================================
echo.

if not exist "models\mao-Q4_K_M.gguf" (
    echo [ERROR] models\mao-Q4_K_M.gguf not found
    pause
    exit /b 1
)
if not exist "tools\llama-server.exe" (
    echo [ERROR] tools\llama-server.exe not found
    pause
    exit /b 1
)

echo [1/3] starting llama-server on 8080 ...
start "mao-server" /min tools\llama-server.exe ^
  -m models\mao-Q4_K_M.gguf ^
  --jinja ^
  --chat-template-file chat_template.jinja ^
  -c 2048 ^
  -t 4 ^
  --repeat-penalty 1.35 ^
  --repeat-last-n 256 ^
  --top-k 40 ^
  --top-p 0.95 ^
  --min-p 0.05 ^
  --host 127.0.0.1 ^
  --port 8080

echo [2/3] waiting for model to load ...
timeout /t 12 /nobreak >nul

echo [3/3] starting proxy on 8081 ...
start "mao-proxy" /min python src\deploy\proxy_system.py

timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo   READY    http://127.0.0.1:8081
echo ============================================================
echo.
pause
