"""Environment audit for reproducible CUDA experiments."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

import torch


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def audit_environment() -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": [properties.major, properties.minor],
                }
            )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "nvidia_smi": _command(["nvidia-smi"]),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE")
            if os.environ.get(key) is not None
        },
    }
