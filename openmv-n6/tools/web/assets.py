"""Load browser resources relative to this package, independent of the CWD."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def read_bytes(name: str) -> bytes:
    return (_ROOT / name).read_bytes()


def read_text(name: str) -> str:
    return (_ROOT / name).read_text(encoding="utf-8")
