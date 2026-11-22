"""Write results/env.json: hardware, OS, toolchain and the llama.cpp commit."""

import json
import os
import platform
import shutil
import subprocess
import datetime

import psutil

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def ps(expr):
    return run(["powershell", "-NoProfile", "-Command", expr])


def main():
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"]).split(", ")
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(WORK)
    repo = os.path.join(WORK, "llama.cpp")
    nvcc = run([os.path.join(CUDA, "bin", "nvcc.exe"), "--version"])
    env = {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": {
            "name": gpu[0],
            "vram_total_mib": int(gpu[1]),
            "vram_used_at_record_mib": int(gpu[2]),
            "driver_version": gpu[3],
        },
        "cuda_toolkit": {
            "used_for_build": next((l.split("release ")[1] for l in nvcc.splitlines() if "release" in l), nvcc),
            "path": CUDA,
        },
        "cpu": {
            "name": ps("(Get-CimInstance Win32_Processor).Name"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gib": round(vm.total / 2**30, 1),
            "available_at_record_gib": round(vm.available / 2**30, 1),
            "modules": ps("Get-CimInstance Win32_PhysicalMemory | ForEach-Object { '{0} GB {1} MHz' -f ($_.Capacity/1GB), $_.ConfiguredClockSpeed }").splitlines(),
        },
        "disk": {
            "work_drive": os.path.splitdrive(WORK)[0],
            "free_gib": round(disk.free / 2**30, 1),
            "model": ps("(Get-Partition -DriveLetter D | Get-Disk).FriendlyName"),
        },
        "os": {
            "name": ps("(Get-CimInstance Win32_OperatingSystem).Caption"),
            "version": platform.version(),
            "build": ps("(Get-CimInstance Win32_OperatingSystem).BuildNumber"),
        },
        "compiler": ps("(Get-ItemProperty 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\*' | Sort-Object Name | Select-Object -Last 1).Name"),
        "llama_cpp": {
            "repo": "https://github.com/ggml-org/llama.cpp",
            "commit": run(["git", "-C", repo, "log", "-1", "--format=%H"]),
            "commit_date": run(["git", "-C", repo, "log", "-1", "--format=%cI"]),
        },
        "model": {
            "repo": "ggml-org/Inkling-2-Mini-GGUF",
            "file": "Inkling-2-Mini-IQ4_NL.gguf",
        },
        "python": platform.python_version(),
    }
    model = os.path.join(WORK, "models", env["model"]["file"])
    if os.path.isfile(model):
        env["model"]["bytes"] = os.path.getsize(model)
    out = os.path.join(WORK, "results", "env.json")
    json.dump(env, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(env, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
