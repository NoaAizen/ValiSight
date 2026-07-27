#!/usr/bin/env python3
"""Deterministic physics linter for core/*.py.

Fast, deterministic, NO model call. Runs as a PostToolUse hook after Edit/Write
on a .py file under core/, and is also usable straight from the CLI:

    python scripts/physics_lint.py core/physics_invariants/velocity.py

It checks three things that CLAUDE.md says have burned a reviewer before:

  1. Range power law     — an inverse-square (`** 2`) exponent in an expression
                           that mixes range and intensity/power. A monostatic
                           point target is 1/R**4 (trap #1).
  2. Magic constants     — an empirical float literal used directly inside a
                           multiply/divide, instead of a named UPPER_CASE
                           constant. Physics scale factors must be named.
  3. Geoid guard         — a file that touches geoid / EGM2008 / height
                           conversion without routing through a live-guard
                           (`_assert_geoid_live` / `geoid_is_live`), trap #7.

Comments and string/docstring bodies are invisible to the linter (it works on
the token stream), so prose that mentions the physics never trips it.

Exit codes: 0 = clean, 2 = at least one violation (blocks the tool call and
feeds the reasons back to Claude). Anything that is not a core/*.py file exits 0.
"""
import io
import json
import os
import re
import sys
import tokenize

# --- rule vocabulary ---------------------------------------------------------

# Structural factors from the governing equations — never "magic".
STRUCTURAL = {0.0, 1.0, 2.0, 4.0, 0.5}

RANGE_NAMES = {"range", "range_m", "rng", "r", "r_m", "dist", "distance",
               "rad_range"}
INTENSITY_NAMES = {"intensity", "power", "amp", "amplitude", "rcs", "snr",
                   "snr_lin", "signal", "falloff", "mag", "magnitude", "p_r",
                   "pr", "returnpower", "return_power"}

GEOID_TRIGGER_SUBSTR = ("geoid", "egm2008", "undulation", "orthometric",
                        "ellipsoidal_height", "geoid_height", "pyproj",
                        "geoidgrids")
# The live-guard itself, plus the sanctioned constructors that route through it
# (they call geoid_is_live internally), so a module that builds its geoid edge
# via them — or merely re-exports them — is correctly guarded.
GEOID_GUARD_NAMES = {"_assert_geoid_live", "geoid_is_live",
                     "register_geoid_edge", "make_geoid_transforms"}

CONST_DEF_RE = re.compile(r"^\s*[A-Z_][A-Z0-9_]*\s*(?::[^=]+)?=")


def _is_power_of_ten(value):
    """Unit-conversion literals (1e3, 1e6, 1e-3, ...) are not magic."""
    if value <= 0:
        return False
    v = value
    while v >= 10 and v == int(v):
        v /= 10
    if v == 1:
        return True
    v = value
    while v < 1:
        v *= 10
    return v == 1


def _is_float_literal(tok_string):
    return ("." in tok_string) or ("e" in tok_string.lower())


class Finding:
    def __init__(self, line, code, msg):
        self.line, self.code, self.msg = line, code, msg

    def __str__(self):
        return "  line %d [%s] %s" % (self.line, self.code, self.msg)


def lint_source(source):
    """Return a list of Findings for the given source text."""
    findings = []
    lines = source.splitlines()
    const_def_lines = {i + 1 for i, ln in enumerate(lines)
                       if CONST_DEF_RE.match(ln)}

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return [Finding(getattr(exc, "lineno", 0) or 0, "parse",
                        "could not tokenize file: %r" % (exc,))]

    # Significant tokens only (drop layout/comments/strings for adjacency work).
    sig = [t for t in toks if t.type in (tokenize.NAME, tokenize.NUMBER,
                                         tokenize.OP)]

    # Names present per source line (for the range/intensity co-occurrence rule).
    names_by_line = {}
    all_names = set()
    for t in sig:
        if t.type == tokenize.NAME:
            names_by_line.setdefault(t.start[0], set()).add(t.string.lower())
            all_names.add(t.string)

    for idx, t in enumerate(sig):
        # --- rule 1: inverse-square in a range/intensity expression ----------
        if t.type == tokenize.OP and t.string == "**":
            nxt = sig[idx + 1] if idx + 1 < len(sig) else None
            if nxt is not None and nxt.type == tokenize.NUMBER \
                    and nxt.string.strip() == "2":
                line_names = names_by_line.get(t.start[0], set())
                if (line_names & RANGE_NAMES) and (line_names & INTENSITY_NAMES):
                    findings.append(Finding(
                        t.start[0], "range-power",
                        "'** 2' in a range/intensity expression - a monostatic "
                        "point target falls off as 1/R**4, not 1/R**2 "
                        "(CLAUDE.md trap #1)."))

        # --- rule 2: magic float literal inside a multiply/divide ------------
        if t.type == tokenize.NUMBER and _is_float_literal(t.string):
            if t.start[0] in const_def_lines:
                continue
            try:
                value = float(t.string)
            except ValueError:
                continue
            if value in STRUCTURAL or _is_power_of_ten(value):
                continue
            prev_tok = sig[idx - 1] if idx > 0 else None
            next_tok = sig[idx + 1] if idx + 1 < len(sig) else None
            adjacent_ops = {p.string for p in (prev_tok, next_tok)
                            if p is not None and p.type == tokenize.OP}
            if adjacent_ops & {"*", "/", "**"}:
                findings.append(Finding(
                    t.start[0], "magic-const",
                    "magic constant %s inside a physics formula - name it as an "
                    "UPPER_CASE module constant instead of inlining it."
                    % t.string))

    # --- rule 3: geoid / height conversion without a live-guard ---------------
    lowered_names = {n.lower() for n in all_names}
    touches_geoid = any(sub in n for n in lowered_names
                        for sub in GEOID_TRIGGER_SUBSTR)
    has_guard = bool(all_names & GEOID_GUARD_NAMES) or any(
        g in n for n in lowered_names for g in GEOID_GUARD_NAMES)
    if touches_geoid and not has_guard:
        findings.append(Finding(
            0, "geoid-guard",
            "file touches geoid / EGM2008 / height conversion but never routes "
            "through a live-guard (_assert_geoid_live / geoid_is_live). pyproj "
            "returns unchanged heights when the grid is missing (CLAUDE.md "
            "trap #7)."))

    return findings


def _resolve_target():
    """File path to lint: argv[1], else the hook's tool_input.file_path."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    data = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not data.strip():
        return None
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    return (payload.get("tool_input") or {}).get("file_path")


def _under_core_py(path):
    norm = path.replace("\\", "/")
    return norm.endswith(".py") and re.search(r"(^|/)core/", norm) is not None


def main():
    target = _resolve_target()
    if not target or not _under_core_py(target) or not os.path.isfile(target):
        return 0  # not our concern -> never block

    with io.open(target, encoding="utf-8") as fh:
        source = fh.read()

    findings = lint_source(source)
    if not findings:
        return 0

    sys.stderr.write("physics_lint: %d violation(s) in %s\n"
                     % (len(findings), target))
    for f in findings:
        sys.stderr.write(str(f) + "\n")
    sys.stderr.write("Fix these before continuing (CLAUDE.md physics traps).\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
