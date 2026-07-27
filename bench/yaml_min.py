"""A tiny, deterministic YAML reader/writer for timing_budgets.yaml only.

PyYAML is not a dependency of this project, and the budgets file is a fixed,
simple shape: a top-level ``stages:`` map of stage -> map of scalar keys. This
module parses exactly that (nested maps, scalars, null/true/false/int/float/
quoted or bare strings, ``#`` comments) and dumps it back. It is NOT a general
YAML implementation and does not try to be.
"""


def _strip_comment(line):
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _scalar(text):
    t = text.strip()
    if (len(t) >= 2) and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
    low = t.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _parse_block(lines, i, indent):
    result = {}
    while i < len(lines):
        ind, content = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise ValueError("unexpected indentation: %r" % content)
        key, sep, val = content.partition(":")
        if not sep:
            raise ValueError("expected 'key: value', got %r" % content)
        key = key.strip()
        val = val.strip()
        if val == "":
            if i + 1 < len(lines) and lines[i + 1][0] > indent:
                child_indent = lines[i + 1][0]
                child, i = _parse_block(lines, i + 1, child_indent)
                result[key] = child
            else:
                result[key] = None
                i += 1
        else:
            result[key] = _scalar(val)
            i += 1
    return result, i


def load_yaml(text):
    """Parse the budgets document text into nested dicts."""
    lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if stripped.strip() == "":
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}
    root, _ = _parse_block(lines, 0, 0)
    return root


def _dump_scalar(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        # quote strings that could be misread as another type
        if value == "" or value.lower() in ("null", "true", "false") \
                or value.strip() != value:
            return '"%s"' % value
        return value
    return repr(value)


def dump_yaml(data):
    """Dump ``{"stages": {name: {k: v}}}`` back to text (best-effort, for our
    schema)."""
    out = []
    stages = (data or {}).get("stages") or {}
    out.append("stages:")
    for name, spec in stages.items():
        out.append("  %s:" % name)
        for k, v in (spec or {}).items():
            out.append("    %s: %s" % (k, _dump_scalar(v)))
    return "\n".join(out) + "\n"
