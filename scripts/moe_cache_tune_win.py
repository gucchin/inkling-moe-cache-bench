"""Run upstream scripts/moe-cache-tune.py on Windows without changing it.

The upstream script stops the server with proc.send_signal(SIGINT). Python on Windows rejects
that signal, and without a clean exit the server does not log the cache statistics the tuner
reads. This wrapper swaps the Popen class the tuner uses: the server gets its own hidden
console and SIGINT becomes a real Ctrl+C on that console. The tuner logic is untouched.

  python scripts/moe_cache_tune_win.py --markdown --out runs/tune -- <llama-server command>
"""

import importlib.util
import os
import signal
import subprocess
import types

from llsrv import CUDA_BIN, WORK, _hidden, send_ctrl_c

UPSTREAM = os.path.join(WORK, "llama.cpp", "scripts", "moe-cache-tune.py")


class WinPopen(subprocess.Popen):
    def __init__(self, args, **kw):
        env = dict(kw.pop("env", None) or os.environ)
        env["PATH"] = CUDA_BIN + os.pathsep + env["PATH"]
        super().__init__(args, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=_hidden(), **kw)

    def send_signal(self, sig):
        if sig == signal.SIGINT:
            if self.poll() is None:
                send_ctrl_c(self.pid)
            return
        super().send_signal(sig)


def main():
    spec = importlib.util.spec_from_file_location("moe_cache_tune", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.subprocess = types.SimpleNamespace(Popen=WinPopen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    mod.main()


if __name__ == "__main__":
    main()
