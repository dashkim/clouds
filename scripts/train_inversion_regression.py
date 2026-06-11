#!/usr/bin/env python3
"""Backward-compatible entry point — see models/ridge/train.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ridge.train import main

if __name__ == "__main__":
    sys.exit(main())
