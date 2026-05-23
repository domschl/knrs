from __future__ import annotations

import socket
import platform
import subprocess
import shutil
import re
from pathlib import Path


def get_cpu_info() -> str:
    """Detect the CPU model name in a cross-platform manner."""
    system = platform.system()
    if system == "Darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if brand:
                return brand
        except Exception:
            pass
        return platform.processor() or "Apple Silicon"

    elif system == "Linux":
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            try:
                content = cpuinfo_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    if line.startswith("model name") or line.startswith("Processor"):
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            return parts[1].strip()
            except Exception:
                pass
        return platform.processor() or "Generic Linux CPU"

    return platform.processor() or "Unknown CPU"


def get_accelerator_info() -> str:
    """Detect GPU/Accelerator hardware (MPS, CUDA, Intel XPU, or CPU fallback)."""
    system = platform.system()
    
    # 1. Check for CUDA via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            gpu_name = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            if gpu_name:
                return f"CUDA ({gpu_name})"
        except Exception:
            pass
        return "CUDA (Unknown GPU)"

    # 2. Check for MPS (Apple Silicon GPU)
    if system == "Darwin":
        machine = platform.machine()
        if machine == "arm64" or "Apple" in get_cpu_info():
            return "MPS (Apple Silicon GPU)"

    # 3. Check for Intel XPU (Intel GPU drivers/libraries)
    intel_gpu = False
    if system == "Linux":
        # Check lspci for Intel graphics controllers
        lspci = shutil.which("lspci")
        if lspci:
            try:
                pci_devices = subprocess.check_output([lspci], text=True)
                if "VGA compatible controller: Intel" in pci_devices or "Display controller: Intel" in pci_devices:
                    intel_gpu = True
            except Exception:
                pass
    if intel_gpu:
        return "Intel XPU/GPU"

    return "CPU Only"


def get_system_info() -> dict[str, str]:
    """Retrieve hostname, OS, CPU, and Accelerator information."""
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": get_cpu_info(),
        "accelerator": get_accelerator_info(),
    }


if __name__ == "__main__":
    info = get_system_info()
    for k, v in info.items():
        print(f"{k.capitalize()}: {v}")
