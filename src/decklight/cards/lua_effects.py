"""Per-effect mechanics extracted deterministically from ygopro card Lua scripts.

Phase 2, layer 1 of the card-knowledge stack: one :class:`EffectRecord` per
``Effect.CreateEffect`` call in a card script, with type/event/property/range/
category/count-limit fields evaluated against ``constant.lua``. No LLM involved.

Only value expressions are evaluated (integers, constant names, ``id``, and the
``|``, ``+``, ``-``, ``<<``, ``>>``, ``&``, ``~`` operators with parentheses).
Anything else (function calls, field references like ``s.x``) is recorded in
``EffectRecord.unresolved`` instead of raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTANT_LUA = REPO_ROOT / "third_party" / "CardScripts" / "constant.lua"
OFFICIAL_DIR = REPO_ROOT / "third_party" / "CardScripts" / "official"


@dataclass
class EffectRecord:
    """One ``Effect.CreateEffect`` (or ``:Clone()``) record from a card script."""

    var: str  # Lua variable name, e.g. "e1"
    type: int = 0  # SetType value, evaluated
    code: int = 0  # SetCode value (event or effect code), evaluated; 0 if absent/unknown
    property: int = 0  # SetProperty flags (EFFECT_FLAG_*), first argument only
    range: int = 0  # SetRange (LOCATION_*)
    category: int = 0  # SetCategory (CATEGORY_*)
    # SetCountLimit args evaluated: (n,) or (n, id_or_flags...); `id` -> card code.
    count_limit: tuple[int, ...] | None = None
    description: int | None = None  # n from SetDescription(aux.Stringid(id,n)); None otherwise
    has_condition: bool = False
    has_cost: bool = False
    has_target: bool = False
    has_operation: bool = False
    cloned_from: str | None = None  # "e1" for `local e2=e1:Clone()`
    registered: bool = False  # passed to c:RegisterEffect / Duel.RegisterEffect
    unresolved: list[str] = field(default_factory=list)  # e.g. "SetCode:s.custom"


# ---------------------------------------------------------------------------
# comment stripping + string masking
# ---------------------------------------------------------------------------

_LONG_OPEN_RE = re.compile(r"\[(=*)\[")


def _strip_comments(text: str) -> str:
    """Remove Lua ``--`` line comments and ``--[[ ... ]]`` block comments.

    ``--`` inside string literals is left alone.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            out.append(ch)
            i += 1
            while i < n:
                c = text[i]
                out.append(c)
                i += 1
                if c == "\\" and i < n:
                    out.append(text[i])
                    i += 1
                elif c == ch:
                    break
        elif ch == "-" and i + 1 < n and text[i + 1] == "-":
            m = _LONG_OPEN_RE.match(text, i + 2)
            if m:  # long/block comment
                close = "]" + m.group(1) + "]"
                j = text.find(close, m.end())
                i = n if j < 0 else j + len(close)
                out.append(" ")
            else:  # line comment: skip to (not including) the newline
                j = text.find("\n", i)
                i = n if j < 0 else j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _mask_strings(text: str) -> str:
    """Blank out string-literal contents (same length) so scans skip over them."""
    chars = list(text)
    i, n = 0, len(text)
    while i < n:
        ch = chars[i]
        if ch in "'\"":
            i += 1
            while i < n:
                if chars[i] == "\\" and i + 1 < n:
                    chars[i] = chars[i + 1] = " "
                    i += 2
                elif chars[i] == ch:
                    break
                else:
                    chars[i] = " "
                    i += 1
        elif ch == "[":
            m = _LONG_OPEN_RE.match(text, i)
            if m:
                close = "]" + m.group(1) + "]"
                j = text.find(close, m.end())
                j = n if j < 0 else j
                for k in range(i, j):
                    if chars[k] != "\n":
                        chars[k] = " "
                i = j
                continue
        i += 1
    return "".join(chars)


# ---------------------------------------------------------------------------
# value-expression evaluator (tokenizer + recursive descent, Lua precedence)
# ---------------------------------------------------------------------------


class _EvalError(ValueError):
    pass


_TOKEN_RE = re.compile(
    r"(?P<num>0[xX][0-9a-fA-F]+|[0-9]+)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><<|>>|[|+&~()\-+])"
    r"|(?P<space>\s+)"
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        m = _TOKEN_RE.match(text, i)
        if m is None:
            raise _EvalError(f"bad character {text[i]!r} in {text!r}")
        i = m.end()
        kind = m.lastgroup or "space"
        if kind != "space":
            toks.append((kind, m.group()))
    return toks


class _ExprParser:
    """``|`` < xor ``~`` < ``&`` < ``<<``/``>>`` < ``+``/``-`` < unary < primary."""

    def __init__(self, toks: list[tuple[str, str]], env: dict[str, int]) -> None:
        self.toks = toks
        self.pos = 0
        self.env = env

    def parse(self) -> int:
        if not self.toks:
            raise _EvalError("empty expression")
        v = self._or()
        if self.pos != len(self.toks):
            raise _EvalError(f"trailing tokens in {self.toks!r}")
        return v

    def _peek(self) -> str | None:
        return self.toks[self.pos][1] if self.pos < len(self.toks) else None

    def _next(self) -> str:
        tok = self.toks[self.pos][1]
        self.pos += 1
        return tok

    def _or(self) -> int:
        v = self._xor()
        while self._peek() == "|":
            self._next()
            v |= self._xor()
        return v

    def _xor(self) -> int:
        v = self._and()
        while self._peek() == "~":
            self._next()
            v ^= self._and()
        return v

    def _and(self) -> int:
        v = self._shift()
        while self._peek() == "&":
            self._next()
            v &= self._shift()
        return v

    def _shift(self) -> int:
        v = self._add()
        while self._peek() in ("<<", ">>"):
            op = self._next()
            rhs = self._add()
            v = v << rhs if op == "<<" else v >> rhs
        return v

    def _add(self) -> int:
        v = self._unary()
        while self._peek() in ("+", "-"):
            op = self._next()
            rhs = self._unary()
            v = v + rhs if op == "+" else v - rhs
        return v

    def _unary(self) -> int:
        ops = []
        while self._peek() in ("-", "~", "+"):
            ops.append(self._next())
        v = self._primary()
        for op in reversed(ops):
            if op == "-":
                v = -v
            elif op == "~":
                v = ~v
        return v

    def _primary(self) -> int:
        tok = self._next() if self.pos < len(self.toks) else None
        if tok is None:
            raise _EvalError("unexpected end of expression")
        kind = self.toks[self.pos - 1][0]
        if kind == "num":
            return int(tok, 16 if tok[:2] in ("0x", "0X") else 10)
        if kind == "ident":
            if tok not in self.env:
                raise _EvalError(f"unknown name {tok!r}")
            return self.env[tok]
        if tok == "(":
            v = self._or()
            if self._peek() != ")":
                raise _EvalError("missing closing paren")
            self._next()
            return v
        raise _EvalError(f"unexpected token {tok!r}")


def _eval_expr(text: str, env: dict[str, int]) -> int:
    return _ExprParser(_tokenize(text), env).parse()


def _depth(text: str) -> int:
    d = 0
    for ch in text:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
    return d


def _split_top(text: str) -> list[str]:
    """Split on top-level commas (strings already masked, comments stripped)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


# ---------------------------------------------------------------------------
# constant.lua
# ---------------------------------------------------------------------------

_CONST_RE = re.compile(r"(?:local\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def load_constants(path: str | Path = CONSTANT_LUA) -> dict[str, int]:
    """Evaluate ``constant.lua`` with the safe expression evaluator (no ``eval``)."""
    text = _mask_strings(_strip_comments(Path(path).read_text(encoding="utf-8")))
    env: dict[str, int] = {}
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        buf = f"{buf} {line}" if buf else line
        if _depth(buf) != 0:  # multi-line value: keep accumulating
            continue
        m = _CONST_RE.match(buf)
        buf = ""
        if m is None:
            continue
        try:
            env[m.group(1)] = _eval_expr(m.group(2), env)
        except _EvalError:
            continue  # e.g. string values or names defined in other LoadScript files
    return env


# ---------------------------------------------------------------------------
# card scripts
# ---------------------------------------------------------------------------

_SCAN_RE = re.compile(
    r"(?P<create>(?:local\s+)?(?P<cvar>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*Effect\.CreateEffect\s*\()"
    r"|(?P<clone>(?:local\s+)?(?P<clvar>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<src>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*Clone\s*\(\s*\))"
    r"|(?P<method>(?P<mvar>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<meth>[A-Za-z_][A-Za-z0-9_]*)\s*\()"
    r"|(?P<colonreg>:\s*RegisterEffect\s*\()"
    r"|(?P<duelreg>Duel\s*\.\s*RegisterEffect\s*\()"
    r"|(?P<proc>(?P<plib>[A-Za-z_][A-Za-z0-9_]*)\.(?P<pfunc>AddProc\w*)\s*\()"
    r"|(?P<localshadow>local\s+(?P<shvar>[A-Za-z_][A-Za-z0-9_]*)\s*=)"
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_STRINGID_RE = re.compile(r"aux\.Stringid\(\s*(?:id|[0-9]+)\s*,\s*([0-9]+)\s*\)$")


def _balanced(text: str, open_idx: int) -> tuple[str, int]:
    """Return (inner text, index just past closer) for the paren at ``open_idx``."""
    depth, i, n = 0, open_idx, len(text)
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
        i += 1
    raise _EvalError("unbalanced parentheses")


def _eval_first(args: list[str], env: dict[str, int]) -> int:
    if not args or not args[0].strip():
        raise _EvalError("missing argument")
    return _eval_expr(args[0], env)


def _apply_setter(rec: EffectRecord, name: str, raw: str, env: dict[str, int]) -> None:
    args = _split_top(raw)
    try:
        if name == "SetType":
            rec.type = _eval_first(args, env)
        elif name == "SetCode":
            rec.code = _eval_first(args, env)
        elif name == "SetProperty":
            rec.property = _eval_first(args, env)  # first argument only
        elif name == "SetRange":
            rec.range = _eval_first(args, env)
        elif name == "SetCategory":
            rec.category = _eval_first(args, env)
        elif name == "SetCountLimit":
            rec.count_limit = _eval_count_limit(args, env)
        elif name == "SetDescription":
            m = _STRINGID_RE.match(args[0].strip()) if args else None
            if m is None:
                if args:
                    _eval_expr(args[0], env)  # evaluable but not a string id: still None
                rec.description = None
            else:
                rec.description = int(m.group(1))
        elif name == "SetCondition":
            rec.has_condition = True
        elif name == "SetCost":
            rec.has_cost = True
        elif name == "SetTarget":
            rec.has_target = True
        elif name == "SetOperation":
            rec.has_operation = True
    except _EvalError:
        if name == "SetCountLimit":
            tag = f"{name}:{raw.strip()}"
        else:
            tag = f"{name}:{args[0].strip() if args else ''}"
        rec.unresolved.append(tag)


def _eval_count_limit(args: list[str], env: dict[str, int]) -> tuple[int, ...]:
    if not args or not args[0].strip():
        raise _EvalError("missing count")
    out: list[int] = [_eval_expr(args[0], env)]
    for part in args[1:]:
        part = part.strip()
        if part.startswith("{") and part.endswith("}"):
            for sub in _split_top(part[1:-1]):
                if sub.strip():
                    out.append(_eval_expr(sub, env))
        elif part:
            out.append(_eval_expr(part, env))
    return tuple(out)


def parse_script(text: str, code: int, constants: dict[str, int]) -> list[EffectRecord]:
    """Parse one card script into :class:`EffectRecord`s (never raises on input)."""
    env = dict(constants or {})
    env["id"] = code
    masked = _mask_strings(_strip_comments(text))
    records: list[EffectRecord] = []
    active: dict[str, int] = {}  # var name -> index of its latest record

    def mark_registered(varname: str) -> None:
        if _IDENT_RE.match(varname) and varname in active:
            records[active[varname]].registered = True

    pos = 0
    while True:
        m = _SCAN_RE.search(masked, pos)
        if m is None:
            break
        kind = m.lastgroup
        if kind == "localshadow":
            active.pop(m.group("shvar"), None)
            pos = m.end()
        elif kind == "clone":
            src, var = m.group("src"), m.group("clvar")
            base = records[active[src]] if src in active else None
            rec = EffectRecord(
                var=var,
                type=base.type if base else 0,
                code=base.code if base else 0,
                property=base.property if base else 0,
                range=base.range if base else 0,
                category=base.category if base else 0,
                count_limit=base.count_limit if base else None,
                description=base.description if base else None,
                has_condition=base.has_condition if base else False,
                has_cost=base.has_cost if base else False,
                has_target=base.has_target if base else False,
                has_operation=base.has_operation if base else False,
                cloned_from=src,
                unresolved=list(base.unresolved) if base else [],
            )
            records.append(rec)
            active[var] = len(records) - 1
            pos = m.end()
        elif kind == "create":
            var = m.group("cvar")
            records.append(EffectRecord(var=var))
            active[var] = len(records) - 1
            _, pos = _balanced(masked, m.end() - 1)
        elif kind == "proc":
            _, pos = _balanced(masked, m.end() - 1)  # not part of EffectRecord; see procedures()
        else:  # method / colonreg / duelreg
            raw, pos = _balanced(masked, m.end() - 1)
            if kind == "method" and not m.group("meth").startswith("Set"):
                if m.group("meth") == "RegisterEffect":
                    mark_registered(_split_top(raw)[0].strip() if raw.strip() else "")
            elif kind == "method":
                rec = records[active[m.group("mvar")]] if m.group("mvar") in active else None
                if rec is not None:
                    _apply_setter(rec, m.group("meth"), raw, env)
            else:
                parts = _split_top(raw)
                mark_registered(parts[0].strip() if parts else "")
    return records


def script_effects(
    code: int, scripts_dir: str | Path = OFFICIAL_DIR, constants: dict[str, int] | None = None
) -> list[EffectRecord]:
    """Parse ``official/c<code>.lua`` into :class:`EffectRecord`s."""
    if constants is None:
        constants = load_constants()
    text = (Path(scripts_dir) / f"c{code}.lua").read_text(encoding="utf-8")
    return parse_script(text, code, constants)


_PROC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.(AddProc\w*)")


def procedures(text: str) -> list[str]:
    """Names of summon-procedure calls (``Fusion.AddProcMix``, ...) in order."""
    masked = _mask_strings(_strip_comments(text))
    return [f"{m.group(1)}.{m.group(2)}" for m in _PROC_RE.finditer(masked)]
