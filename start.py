"""
TSL Translator — One-Command Start
Starts the FastAPI backend + serves the web frontend.

Usage:
    python start.py                    # Start both backend + frontend
    python start.py --backend-only     # Start only the API
    python start.py --frontend-only    # Start only the web UI
    python start.py --port 8080        # Custom port
"""
import sys
import os
import subprocess
import argparse
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def start_backend(host, port):
    """Start the FastAPI backend."""
    print(f"Starting API server on http://{host}:{port}...")
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", host, "--port", str(port)]
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def start_frontend(port):
    """Start a simple HTTP server for the web frontend."""
    web_dir = PROJECT_ROOT / "web"
    if not web_dir.exists():
        print(f"✗ Web directory not found: {web_dir}")
        return None

    print(f"Starting web frontend on http://localhost:{port}...")
    cmd = [sys.executable, "-m", "http.server", str(port)]
    return subprocess.Popen(cmd, cwd=web_dir)


def main():
    parser = argparse.ArgumentParser(description="TSL Translator Start")
    parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--web-port", type=int, default=3000, help="Web port (default: 3000)")
    parser.add_argument("--backend-only", action="store_true", help="Start only backend")
    parser.add_argument("--frontend-only", action="store_true", help="Start only frontend")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    print("=" * 50)
    print("  TSL Tunisian Sign Language Translator")
    print("=" * 50)

    processes = []

    if not args.frontend_only:
        processes.append(("Backend", start_backend(args.host, args.port)))

    if not args.backend_only:
        fe = start_frontend(args.web_port)
        if fe:
            processes.append(("Frontend", fe))

    if not processes:
        print("Nothing to start!")
        return

    print()
    print("  API docs:  http://localhost:{}/docs".format(args.port))
    if not args.backend_only:
        print("  Web app:   http://localhost:{}".format(args.web_port))
    print()
    print("  Press Ctrl+C to stop")
    print()

    if not args.no_browser and not args.backend_only:
        time.sleep(2)
        webbrowser.open(f"http://localhost:{args.web_port}")

    try:
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for name, proc in processes:
            print(f"  Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done!")


if __name__ == "__main__":
    main()
