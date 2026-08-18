"""Phase 0 — Environment + Verification.

Run: python scripts/verify_env.py
Exit 0 only if every check passes.
"""
import shutil
import sys

MIN_CUDA = (12, 1)
MIN_PYTHON = (3, 11)
MIN_FLASH_ATTN = (2, 5)
MIN_BNB = (0, 43)
MIN_DISK_GB = 500
WARN_DISK_GB = 2000
MIN_RAM_GB = 32


def _parse_version(v: str) -> tuple:
    parts = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) == MIN_PYTHON
    return ok, f"Python {v.major}.{v.minor}.{v.micro} (need 3.11.x)"


def check_torch_install() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} — CUDA not available"
    return True, f"torch {torch.__version__} — CUDA available"


def check_cuda_version() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed, cannot check CUDA"
    if not torch.cuda.is_available():
        return False, "no CUDA device visible"
    cuda_ver = torch.version.cuda
    if cuda_ver is None:
        return False, "torch built without CUDA support"
    ok = _parse_version(cuda_ver) >= MIN_CUDA
    device = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return ok, f"CUDA {cuda_ver} on {device}, {vram_gb:.1f}GB VRAM (need >= {MIN_CUDA[0]}.{MIN_CUDA[1]})"


def check_flash_attn() -> tuple[bool, str]:
    try:
        import flash_attn
    except ImportError:
        return False, "flash_attn not installed"
    ver = _parse_version(flash_attn.__version__)
    ok = ver >= MIN_FLASH_ATTN
    return ok, f"flash_attn {flash_attn.__version__} (need >= {MIN_FLASH_ATTN[0]}.{MIN_FLASH_ATTN[1]})"


def check_bitsandbytes() -> tuple[bool, str]:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False, "bitsandbytes not installed"
    ver = _parse_version(bnb.__version__)
    ok = ver >= MIN_BNB
    return ok, f"bitsandbytes {bnb.__version__} (need >= {MIN_BNB[0]}.{MIN_BNB[1]})"


def check_disk_space(target_path: str = ".") -> tuple[bool, str]:
    free_gb = shutil.disk_usage(target_path).free / (1024**3)
    ok = free_gb >= MIN_DISK_GB
    warn = free_gb < WARN_DISK_GB
    msg = f"{free_gb:.0f}GB free (need >= {MIN_DISK_GB}GB)"
    if ok and warn:
        msg += f" — WARNING: below recommended {WARN_DISK_GB}GB"
    return ok, msg


def check_ram() -> tuple[bool, str]:
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        total_gb = kb / (1024**2)
    ok = total_gb >= MIN_RAM_GB
    return ok, f"{total_gb:.1f}GB RAM (need >= {MIN_RAM_GB}GB)"


def print_full_report() -> None:
    checks = [
        ("Python version", check_python_version),
        ("Torch install", check_torch_install),
        ("CUDA version", check_cuda_version),
        ("Flash Attention", check_flash_attn),
        ("bitsandbytes", check_bitsandbytes),
        ("Disk space", check_disk_space),
        ("System RAM", check_ram),
    ]
    print("=== NEO PHASE 0 — ENVIRONMENT VERIFICATION ===\n")
    all_pass = True
    for name, fn in checks:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"  [{status}] {name:<18} {detail}")
    print()
    if all_pass:
        print("ALL PASS")
    else:
        print("FAILED — fix the above before proceeding to Phase 1")
        sys.exit(1)


if __name__ == "__main__":
    print_full_report()
