"""
TSL Translator — Complete Auto-Setup
Run this script once to set up everything: venv, dependencies, models, web app.

Usage:
    python setup.py
    python setup.py --skip-deps      # skip dependency installation
    python setup.py --skip-models    # skip model download/training
"""
import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd, cwd=None, check=True):
    """Run a command and stream output."""
    print(f"  → {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or PROJECT_ROOT,
        capture_output=False, text=True
    )
    if check and result.returncode != 0:
        print(f"  ✗ Command failed: {cmd}")
        sys.exit(1)
    return result


def check_python():
    """Check Python version."""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"✗ Python 3.10+ required, found {v.major}.{v.minor}")
        sys.exit(1)
    print(f"✓ Python {v.major}.{v.minor}.{v.micro}")


def install_deps():
    """Install pip dependencies."""
    print("\n[2/6] Installing dependencies...")
    run(f"{sys.executable} -m pip install --upgrade pip")
    run(f"{sys.executable} -m pip install -r requirements.txt")


def setup_env():
    """Create .env from .env.example if not exists."""
    print("\n[3/6] Setting up environment...")
    env_path = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    if not env_path.exists() and env_example.exists():
        shutil.copy2(env_example, env_path)
        print("  ✓ Created .env from .env.example")
    elif env_path.exists():
        print("  ✓ .env already exists")
    else:
        print("  ⚠ No .env.example found")


def train_alphabet_model():
    """Train the alphabet model if not already trained."""
    print("\n[4/6] Checking alphabet model...")
    model_path = PROJECT_ROOT / "models" / "alphabet" / "spell_classifier_best.pth"
    if model_path.exists():
        print(f"  ✓ Alphabet model exists at {model_path}")
        return

    processed_dir = PROJECT_ROOT / "data" / "alphabet" / "processed"
    arsl_csv = PROJECT_ROOT / "data" / "alphabet" / "ArSL_dataset.csv"

    if not processed_dir.exists():
        if not arsl_csv.exists():
            print("  ⚠ ArSL dataset not found — alphabet model training skipped")
            print("    Download from: https://zenodo.org/records/18363162")
            return
        print("  → Preparing ArSL dataset...")
        run(f"{sys.executable} scripts/prepare_arsl.py")

    if processed_dir.exists():
        print("  → Training alphabet models...")
        run(f"{sys.executable} scripts/train_alphabet_models.py")


def verify_installation():
    """Verify the installation works."""
    print("\n[5/6] Verifying installation...")
    checks = [
        ("import torch", "PyTorch"),
        ("import cv2", "OpenCV"),
        ("import mediapipe", "MediaPipe"),
        ("import numpy", "NumPy"),
        ("import fastapi", "FastAPI"),
        ("from src.translation.translator import TSLTranslator", "TSL Translator"),
    ]
    for imp, name in checks:
        result = subprocess.run(
            [sys.executable, "-c", imp],
            capture_output=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} — {result.stderr.decode().strip()}")


def print_summary():
    """Print usage summary."""
    print("\n[6/6] Setup complete!")
    print("=" * 50)
    print("  TSL Tunisian Sign Language Translator")
    print("=" * 50)
    print()
    print("  Start the app:")
    print("    python start.py")
    print()
    print("  Or start manually:")
    print("    Backend:  python -m api.main")
    print("    Frontend: cd web && python -m http.server 3000")
    print()
    print("  API docs: http://localhost:8000/docs")
    print("  Web app:  http://localhost:3000")
    print()
    print("  Configure LLM correction:")
    print("    Edit .env and set LLM_API_KEY")
    print("    Or configure via the web UI Settings tab")
    print()


def main():
    parser = argparse.ArgumentParser(description="TSL Translator Setup")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model training")
    args = parser.parse_args()

    print("=" * 50)
    print("  TSL Translator — Auto Setup")
    print("=" * 50)

    print("\n[1/6] Checking Python...")
    check_python()

    if not args.skip_deps:
        install_deps()
    else:
        print("\n[2/6] Skipping dependencies")

    setup_env()

    if not args.skip_models:
        train_alphabet_model()
    else:
        print("\n[4/6] Skipping model training")

    verify_installation()
    print_summary()


if __name__ == "__main__":
    main()
