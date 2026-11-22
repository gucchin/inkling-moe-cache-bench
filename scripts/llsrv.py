"""Start, stop and watch a llama-server process on Windows.

The MoE cache logs its hit rate when the server exits cleanly, so the server must get a real
Ctrl+C. The server runs in its own hidden console and a helper attaches to that console to send it.
"""

import ctypes
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(WORK, "llama.cpp", "build", "bin", "llama-server.exe")
MODEL = os.path.join(WORK, "models", "Inkling-2-Mini-IQ4_NL.gguf")
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"

CTRL_C_HELPER = (
    "import ctypes,sys,time;k=ctypes.windll.kernel32;k.FreeConsole();"
    "ok=k.AttachConsole(int(sys.argv[1]));k.SetConsoleCtrlHandler(None,1);"
    "k.GenerateConsoleCtrlEvent(0,0);time.sleep(2);sys.exit(0 if ok else 3)"
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(port, endpoint, body=None, timeout=3600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout
    return int(out.strip().splitlines()[0])


def gpu_shared_mib():
    """shared (system) memory used by the GPU adapters: grows when dedicated VRAM overflows"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "((Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()
    try:
        return float(out) / 2**20
    except ValueError:
        return float("nan")


def send_ctrl_c(pid):
    return subprocess.run(
        [sys.executable, "-c", CTRL_C_HELPER, str(pid)],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=_hidden(),
    ).returncode


def _hidden():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


class Server:
    """llama-server with a memory sampler. Use as a context manager."""

    def __init__(self, args, log_path, env=None, port=None, sample_s=0.5):
        self.port = port or free_port()
        self.args = [SERVER] + list(args) + ["--host", "127.0.0.1", "--port", str(self.port)]
        self.log_path = log_path
        self.env = dict(os.environ)
        self.env["PATH"] = CUDA_BIN + os.pathsep + self.env["PATH"]
        self.env.update(env or {})
        self.sample_s = sample_s
        self.proc = None
        self.peak = {"vram_total_mib": 0, "rss_mib": 0.0, "private_mib": 0.0}
        self.vram_before_mib = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        p = psutil.Process(self.proc.pid)
        while not self._stop.is_set():
            try:
                mi = p.memory_info()
                self.peak["rss_mib"] = max(self.peak["rss_mib"], mi.rss / 2**20)
                self.peak["private_mib"] = max(self.peak["private_mib"], mi.private / 2**20)
                self.peak["vram_total_mib"] = max(self.peak["vram_total_mib"], gpu_used_mib())
            except (psutil.Error, ValueError, IndexError):
                pass
            self._stop.wait(self.sample_s)

    def reset_peak(self):
        self.peak = {"vram_total_mib": 0, "rss_mib": 0.0, "private_mib": 0.0}

    def start(self, timeout=1800):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.vram_before_mib = gpu_used_mib()
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.args, stdout=log, stderr=subprocess.STDOUT, env=self.env,
            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=_hidden(),
        )
        log.close()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                self._stop.set()
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}), see {self.log_path}")
            try:
                if http(self.port, "/health", timeout=5).get("status") == "ok":
                    return self
            except OSError:
                pass
            if time.time() - t0 > timeout:
                self.stop()
                raise RuntimeError("llama-server did not become ready")
            time.sleep(1)

    def stop(self):
        self._stop.set()
        if self.proc is None or self.proc.poll() is not None:
            return
        send_ctrl_c(self.proc.pid)
        try:
            self.proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def log(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
