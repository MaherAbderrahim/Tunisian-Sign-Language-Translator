"""
Setup All - One-shot script to download data, prepare features, and train models.

Usage:
    python scripts/setup_all.py              # Full setup (downloads + trains)
    python scripts/setup_all.py --skip-download  # Skip download, use existing data
    python scripts/setup_all.py --quick      # Quick mode (synthetic alphabet data)
"""

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, cmd: list, cwd: Path = None) -> bool:
    """Run a setup step and return success."""
    print()
    print("=" * 60)
    print(f"  STEP: {name}")
    print("=" * 60)
    try:
        result = subprocess.run(
            [sys.executable] + cmd,
            cwd=cwd or PROJECT_ROOT,
            check=True,
        )
        print(f"  [OK] {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {name} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full project setup")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing data)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: use synthetic alphabet data")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training")
    args = parser.parse_args()

    print()
    print("#" * 60)
    print("#  TSL TRANSLATOR - FULL SETUP")
    print("#" * 60)

    steps_ok = 0
    steps_fail = 0

    # ── Step 1: Download W7CH TSL dataset ──
    if not args.skip_download:
        if run_step("Download TSL Dataset (W7CH)", ["scripts/download_data.py"]):
            steps_ok += 1
        else:
            steps_fail += 1
            print("  WARNING: Dataset download failed. Check your internet connection.")
            print("  If you already have the data, use --skip-download")

    # ── Step 2: Prepare W7CH data (landmarks extraction) ──
    if run_step("Prepare TSL Dataset (extract landmarks)", ["scripts/prepare_data.py"]):
        steps_ok += 1
    else:
        steps_fail += 1

    # ── Step 3: Prepare alphabet data ──
    if args.quick:
        if run_step("Create Synthetic Alphabet Dataset", 
                     ["scripts/prepare_alphabet_data.py", "--synthetic"]):
            steps_ok += 1
        else:
            steps_fail += 1
    elif not args.skip_download:
        if run_step("Prepare Alphabet Dataset (HaGRID)", 
                     ["scripts/prepare_alphabet_data.py", "--prepare"]):
            steps_ok += 1
        else:
            steps_fail += 1
            print("  Falling back to synthetic alphabet data...")
            if run_step("Create Synthetic Alphabet Dataset (fallback)",
                         ["scripts/prepare_alphabet_data.py", "--synthetic"]):
                steps_ok += 1

    # ── Step 4: Train sign classifier (V1) ──
    if not args.skip_train:
        if run_step("Train Sign Classifier (57 classes)",
                     ["-m", "src.models.train"]):
            steps_ok += 1
        else:
            steps_fail += 1
            print("  WARNING: Sign model training failed.")

    # ── Step 5: Train spell classifier (Version A) ──
    if not args.skip_train:
        if run_step("Train Spell Classifier (38 classes, Version A)",
                     ["scripts/train_spell.py"]):
            steps_ok += 1
        else:
            steps_fail += 1
            print("  WARNING: Spell model training failed.")

    # ── Summary ──
    print()
    print("#" * 60)
    print(f"#  SETUP COMPLETE: {steps_ok} OK, {steps_fail} failed")
    print("#" * 60)
    print()
    if steps_fail == 0:
        print("All models trained. You can now translate videos:")
        print()
        print("  python scripts/translate_video.py <video_path>")
        print()
        print("Examples:")
        print("  python scripts/translate_video.py video.mp4")
        print("  python scripts/translate_video.py video.mp4 --output result.txt")
        print("  python scripts/translate_video.py video.mp4 --lang fr")
    else:
        print("Some steps failed. Check the output above.")
        print("You can retry individual steps:")
        print("  python scripts/download_data.py")
        print("  python scripts/prepare_data.py")
        print("  python scripts/prepare_alphabet_data.py --synthetic")
        print("  python -m src.models.train")
        print("  python scripts/train_spell.py")


if __name__ == "__main__":
    main()
