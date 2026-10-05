"""Hardware / software provenance recorded with every result (rule S0-1)."""

from __future__ import annotations

import platform
import subprocess
from typing import Dict

import torch

from skyflow.data.cache import git_commit


def _cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"], stderr=subprocess.DEVNULL, timeout=20
            ).decode(errors="ignore").strip()
            return out.splitlines()[0] if out else platform.processor()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _nvidia_smi_driver() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip().splitlines()[0]
    except Exception:
        return "unknown"


def env_info(device: torch.device) -> Dict:
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
        "cpu_model": _cpu_model(),
        "gpu_model": None,
        "gpu_memory_gb": None,
        "driver": None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_model"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["driver"] = _nvidia_smi_driver()
    return info
