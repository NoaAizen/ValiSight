#!/usr/bin/env python3
"""Check first-party source syntax without importing code or touching hardware."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True)
    paths = sorted(set(result.stdout.decode().split("\0")) - {""})
    checked = 0
    errors = []
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            if name.endswith((".py", ".py.tmpl")):
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
            elif path.suffix == ".sh":
                subprocess.run(["bash", "-n", str(path)], check=True,
                               capture_output=True, text=True)
            elif path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                # Colab/IPython magics are not Python syntax. Validate the
                # document structure here; bundle tests cover exported code.
                assert notebook["nbformat"] == 4, "expected notebook format 4"
                assert isinstance(notebook["cells"], list), "missing cells"
                for cell in notebook["cells"]:
                    assert cell["cell_type"] in {"code", "markdown", "raw"}
                    assert isinstance(cell["source"], (str, list))
            else:
                continue
            checked += 1
        except (SyntaxError, ValueError, AssertionError, KeyError,
                TypeError, OSError, subprocess.CalledProcessError) as exc:
            detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            errors.append(f"{name}: {detail}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Source check: {checked} files passed, {len(errors)} failed")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
