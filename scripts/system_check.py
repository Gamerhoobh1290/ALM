"""Repeatable local environment inventory for the AdamLM project."""

import os
import platform
import shutil
import subprocess
import sys


def command_version(command, args=("--version",)):
    try:
        result = subprocess.run([command, *args], capture_output=True, text=True, timeout=15)
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        return "NOT_FOUND"


print(f"platform={platform.platform()}")
print(f"python={sys.version.split()[0]} executable={sys.executable}")
print(f"git={command_version('git')}")
for tool in ("code.cmd" if os.name == "nt" else "code", "cmake", "ninja", "docker", "conda", "uv", "poetry"):
    print(f"{tool.removesuffix('.cmd')}={command_version(tool)}")
print(f"nvidia_smi={command_version('nvidia-smi', ('--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader'))}")
for drive in ("C:\\", "D:\\"):
    if os.path.exists(drive):
        usage = shutil.disk_usage(drive)
        print(f"disk_{drive[0]}_free_gib={usage.free/1024**3:.2f}")
