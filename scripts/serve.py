#!/usr/bin/env python
"""Serve the scoring API.

Usage
-----
    python scripts/serve.py                        # http://127.0.0.1:8000
    python scripts/serve.py --port 9000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn  # noqa: E402

from frauddetect.api import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the fraud scoring API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    arguments = parser.parse_args()

    uvicorn.run(
        create_app(),
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
