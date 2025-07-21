import subprocess
import webbrowser
import time
import sys
from pathlib import Path

# 1) Streamlit サーバーを起動
#    stdout/stderr は隠しても可
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# 2) 少し待ってブラウザを開く
time.sleep(3)
url = "http://localhost:8501"
webbrowser.open(url)

# 3) サーバーが終了するまで待機
proc.wait()
