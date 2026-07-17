#!/usr/bin/env python3
"""Capture reproducibility metadata without exposing authentication tokens."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch


def command(*args: str) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


metadata = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "kernel": platform.release(),
    "machine": platform.machine(),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "git_status_porcelain": command("git", "status", "--porcelain"),
    "packages": {
        "vllm": version("vllm"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "flashinfer-python": version("flashinfer-python"),
    },
    "cuda": {
        "torch_runtime": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": command("nvidia-smi"),
    },
}

output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
serialized = json.dumps(metadata, indent=2)
if output:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
else:
    print(serialized)

