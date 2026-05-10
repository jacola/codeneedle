"""Parse a source file and enumerate named functions with enough body lines to test."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


MIN_BODY_LINES = 20
BONUS_CAP = 40  # extra lines past the primary 20 that count toward the "blue" bonus


@dataclass
class FunctionTarget:
    name: str
    start_line: int           # 1-indexed line of first body line (after the opening brace)
    body_lines: list[str]     # body lines starting at start_line, excluding the closing brace line
    language: str = "js"      # "js" or "py" — controls the prompt wording
    source_path: Path | None = None  # which file this came from (for multi-file corpora)

    @property
    def primary_lines(self) -> list[str]:
        return self.body_lines[:MIN_BODY_LINES]

    @property
    def bonus_lines(self) -> list[str]:
        return self.body_lines[MIN_BODY_LINES:MIN_BODY_LINES + BONUS_CAP]


@dataclass
class Source:
    """A combined corpus: one or more files concatenated for a single benchmark run."""
    files: list[Path]
    text: str                       # full text fed to the model
    targets: list[FunctionTarget]
    language: str

    @property
    def display_name(self) -> str:
        if len(self.files) == 1:
            return self.files[0].name
        return f"{len(self.files)} files from {self.files[0].parent}"


def language_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".js", ".mjs", ".cjs"):
        return "js"
    if suffix == ".py":
        return "py"
    if suffix == ".cs":
        return "cs"
    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .js, .mjs, .cjs, .py, .cs")


def extract(path: Path) -> list[FunctionTarget]:
    source = path.read_text()
    lang = language_of(path)
    if lang == "js":
        targets = _extract_js(source)
    elif lang == "cs":
        targets = _extract_cs(source)
    else:
        targets = _extract_py(source)
    for t in targets:
        t.language = lang
        t.source_path = path
    return targets


def load_source_glob(
    directory: Path,
    glob: str,
    limit: int | None = None,
) -> Source:
    """Glob a directory for source files, concatenate them, extract all targets.

    Files are concatenated with comment-marker headers so the model can see file
    boundaries. All files must be the same language. Across files, duplicate
    function names are deduplicated (first occurrence wins) so the prompt is
    unambiguous when looked up by name.
    """
    paths = sorted(p for p in directory.glob(glob) if p.is_file())
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no files match {directory}/{glob}")

    lang = language_of(paths[0])
    for p in paths[1:]:
        if language_of(p) != lang:
            raise ValueError(
                f"mixed languages in glob: {paths[0]} is {lang}, {p} is {language_of(p)}"
            )

    parts: list[str] = []
    targets: list[FunctionTarget] = []
    seen_names: set[str] = set()
    line_offset = 0

    for p in paths:
        text = p.read_text()
        header = _file_header(lang, p)
        parts.append(header)
        parts.append(text)
        if not text.endswith("\n"):
            parts.append("\n")
        parts.append("\n")  # blank line between files

        header_line_count = header.count("\n")
        for t in extract(p):
            if t.name in seen_names:
                # Skip cross-file collisions — prompt would be ambiguous by name.
                continue
            seen_names.add(t.name)
            t.start_line += line_offset + header_line_count
            targets.append(t)

        line_offset += header.count("\n") + text.count("\n") + (0 if text.endswith("\n") else 1) + 1

    combined = "".join(parts)
    return Source(files=paths, text=combined, targets=targets, language=lang)


def _file_header(lang: str, path: Path) -> str:
    marker = "//" if lang in ("js", "cs") else "#"
    return f"{marker} ====== {path} ======\n"


# --- JavaScript ---------------------------------------------------------------


def _extract_js(source: str) -> list[FunctionTarget]:
    import esprima

    try:
        tree = esprima.parseModule(
            source, options={"loc": True, "tolerant": True}
        )
    except Exception:
        tree = esprima.parseScript(
            source, options={"loc": True, "tolerant": True}
        )

    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    def emit(name: str, block) -> None:
        if name in seen:
            return
        brace_line = block.loc.start.line  # line of '{'
        close_line = block.loc.end.line    # line of '}'
        if close_line - brace_line < MIN_BODY_LINES + 1:
            return
        # lines strictly between { and }
        body = lines[brace_line:close_line - 1]
        if len(body) < MIN_BODY_LINES:
            return
        seen.add(name)
        targets.append(
            FunctionTarget(
                name=name,
                start_line=brace_line + 1,
                body_lines=body,
            )
        )

    def hint_for(parent_type: str | None, key: str, parent) -> str | None:
        if parent_type == "VariableDeclarator" and key == "init":
            pid = getattr(parent, "id", None)
            if pid is not None and getattr(pid, "type", None) == "Identifier":
                return pid.name
        elif parent_type == "AssignmentExpression" and key == "right":
            left = getattr(parent, "left", None)
            if left is None:
                return None
            if getattr(left, "type", None) == "Identifier":
                return left.name
            if getattr(left, "type", None) == "MemberExpression":
                prop = getattr(left, "property", None)
                if prop is not None and getattr(prop, "type", None) == "Identifier":
                    return prop.name
        elif parent_type == "Property" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
            if k is not None and getattr(k, "type", None) == "Literal":
                return str(k.value)
        elif parent_type == "MethodDefinition" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
        return None

    def walk(node, name_hint: str | None = None) -> None:
        if node is None or not hasattr(node, "type"):
            return
        t = node.type

        if t == "FunctionDeclaration":
            nm = (node.id.name if getattr(node, "id", None) else None) or name_hint
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, body)
        elif t == "FunctionExpression":
            nm = (
                (node.id.name if getattr(node, "id", None) else None)
                or name_hint
            )
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, body)
        elif t == "ArrowFunctionExpression":
            body = getattr(node, "body", None)
            if name_hint and body is not None and body.type == "BlockStatement":
                emit(name_hint, body)

        # recurse
        for key, val in vars(node).items():
            if key == "loc":
                continue
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, "type"):
                        walk(item, hint_for(t, key, node))
            elif hasattr(val, "type"):
                walk(val, hint_for(t, key, node))

    walk(tree)
    return targets


# --- Python -------------------------------------------------------------------


def _extract_py(source: str) -> list[FunctionTarget]:
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in seen or not node.body:
            continue
        start = node.body[0].lineno
        end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
        body = lines[start - 1:end]
        if len(body) < MIN_BODY_LINES:
            continue
        seen.add(node.name)
        targets.append(
            FunctionTarget(name=node.name, start_line=start, body_lines=body)
        )
    return targets


# --- C# -----------------------------------------------------------------------

import re

# C# keywords / control-flow that look like method calls syntactically
# (e.g. `foreach (var x in xs)`) but must not be extracted as methods.
_CS_KEYWORDS = frozenset({
    "foreach", "for", "while", "do", "if", "else", "switch", "case",
    "lock", "using", "try", "catch", "finally", "throw", "return",
    "yield", "await", "checked", "unchecked", "fixed", "typeof",
    "sizeof", "nameof", "default", "delegate", "stackalloc", "when",
})

_CS_METHOD_RE = re.compile(
    r'''
    ^\s*                                     # leading whitespace
    (?:(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|new|partial|unsafe|extern)\s+)*  # modifiers
    [\w<>\[\],\s\?]+?\s+                       # return type (greedy-minimal)
    (\w+)                                     # method name (captured)
    \s*\([^)]*\)                              # parameter list
    (?:\s*where\s+[^{]+)?                     # optional generic constraints
    \s*$                                      # end of line (opening brace on next line or same line)
    ''',
    re.VERBOSE | re.MULTILINE,
)


def _extract_cs(source: str) -> list[FunctionTarget]:
    """Extract C# methods by regex + brace matching.

    C# conventions: method signature on one line, opening '{' on same or next line.
    We find the opening brace, then match braces to locate the closing one.
    """
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    for match in _CS_METHOD_RE.finditer(source):
        name = match.group(1)
        if name in _CS_KEYWORDS or name in seen:
            continue

        # Find which line number contains the match
        match_line_idx = source[:match.start()].count("\n")

        # Find the opening brace — could be on the same line or next few lines
        brace_line_idx = None
        for scan in range(match_line_idx, min(match_line_idx + 3, len(lines))):
            if "{" in lines[scan]:
                brace_line_idx = scan
                break

        if brace_line_idx is None:
            continue

        # Brace-match to find the end of the method body
        depth = 0
        close_line_idx = None
        for i in range(brace_line_idx, len(lines)):
            for ch in lines[i]:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        close_line_idx = i
                        break
            if close_line_idx is not None:
                break

        if close_line_idx is None:
            continue

        # Body lines are between the opening brace line and closing brace line
        body = lines[brace_line_idx + 1:close_line_idx]
        if len(body) < MIN_BODY_LINES:
            continue

        seen.add(name)
        targets.append(
            FunctionTarget(
                name=name,
                start_line=brace_line_idx + 2,  # 1-indexed, first line after '{'
                body_lines=body,
            )
        )

    return targets


# --- Sampling -----------------------------------------------------------------


def stratified_sample(
    targets: list[FunctionTarget],
    total_lines: int,
    k: int = 16,
    seed: int = 42,
) -> list[FunctionTarget]:
    """Sample k targets spread across file position — tests recall at all depths, not just the tail."""
    if len(targets) <= k:
        return list(targets)
    targets = sorted(targets, key=lambda t: t.start_line)
    rng = random.Random(seed)
    buckets: list[list[FunctionTarget]] = [[] for _ in range(k)]
    for t in targets:
        idx = min(k - 1, (t.start_line * k) // max(1, total_lines))
        buckets[idx].append(t)
    chosen: list[FunctionTarget] = []
    for b in buckets:
        if b:
            chosen.append(rng.choice(b))
    chosen_names = {t.name for t in chosen}
    pool = [t for t in targets if t.name not in chosen_names]
    rng.shuffle(pool)
    while len(chosen) < k and pool:
        chosen.append(pool.pop())
    return chosen[:k]
