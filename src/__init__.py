from pathlib import Path

# Dynamically load version from top-level VERSION file
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
__version__ = VERSION_FILE.read_text(encoding="utf-8").strip()
