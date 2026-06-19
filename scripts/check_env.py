"""Environment sanity check.

Run after installing requirements:

    python scripts/check_env.py

Reports the Python version, key library versions, and whether PyTorch can see a
CUDA GPU. Exits non-zero if a core import fails.
"""

import importlib
import platform
import sys

CORE_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "transformers",
    "datasets",
    "matplotlib",
    "seaborn",
]


def main() -> int:
    print(f"Python : {platform.python_version()}  ({sys.executable})")

    ok = True
    for name in CORE_PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok]   {name:<12} {version}")
        except ImportError as exc:
            ok = False
            print(f"  [MISS] {name:<12} ({exc})")

    # PyTorch GPU visibility
    try:
        import torch

        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"PyTorch CUDA available: {torch.cuda.device_count()} GPU(s) -> {names}")
        else:
            print("PyTorch CUDA not available (CPU-only). For GPU see "
                  "https://pytorch.org/get-started/locally/")
    except ImportError:
        pass

    print("\nEnvironment OK." if ok else "\nEnvironment INCOMPLETE — see [MISS] above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
