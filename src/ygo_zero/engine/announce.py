"""Which card codes satisfy an ANNOUNCE_CARD filter. Port of `is_declarable` in `playerop.cpp`."""

from collections.abc import Iterable, Sequence

from ygo_zero.engine import constants as C
from ygo_zero.engine.resources import CardData


def _trunc_div(a: int, b: int) -> int:
    if b == 0:
        return 0  # undefined in C; never emitted by real scripts
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


_BINARY = {
    C.OPCODE_ADD: lambda a, b: a + b,
    C.OPCODE_SUB: lambda a, b: a - b,
    C.OPCODE_MUL: lambda a, b: a * b,
    C.OPCODE_DIV: _trunc_div,
    C.OPCODE_AND: lambda a, b: int(bool(a) and bool(b)),
    C.OPCODE_OR: lambda a, b: int(bool(a) or bool(b)),
    C.OPCODE_BAND: lambda a, b: a & b,
    C.OPCODE_BOR: lambda a, b: a | b,
    C.OPCODE_BXOR: lambda a, b: a ^ b,
    C.OPCODE_LSHIFT: lambda a, b: a << b,
    C.OPCODE_RSHIFT: lambda a, b: a >> b,
}
_UNARY = {
    C.OPCODE_NEG: lambda cd, v: -v,
    C.OPCODE_NOT: lambda cd, v: int(not v),
    C.OPCODE_BNOT: lambda cd, v: ~v,
    C.OPCODE_ISCODE: lambda cd, v: int(cd.code == (v & 0xFFFFFFFF)),
    C.OPCODE_ISTYPE: lambda cd, v: cd.type & v,
    C.OPCODE_ISRACE: lambda cd, v: cd.race & v,
    C.OPCODE_ISATTRIBUTE: lambda cd, v: cd.attribute & v,
}
_GET = {
    C.OPCODE_GETCODE: lambda cd: cd.code,
    C.OPCODE_GETTYPE: lambda cd: cd.type,
    C.OPCODE_GETRACE: lambda cd: cd.race,
    C.OPCODE_GETATTRIBUTE: lambda cd: cd.attribute,
}


def _is_setcard(cd: CardData, set_code: int) -> int:
    settype, subtype = set_code & 0xFFF, set_code & 0xF000
    return int(any((sc & 0xFFF) == settype and (sc & 0xF000 & subtype) == subtype for sc in cd.setcodes))


def is_declarable(cd: CardData, opcodes: Sequence[int]) -> bool:
    stack: list[int] = []
    alias = token = False
    for op in opcodes:
        if op in _BINARY:
            if len(stack) >= 2:
                rhs, lhs = stack.pop(), stack.pop()
                stack.append(_BINARY[op](lhs, rhs))
        elif op in _UNARY:
            if stack:
                stack.append(_UNARY[op](cd, stack.pop()))
        elif op in _GET:
            stack.append(_GET[op](cd))
        elif op == C.OPCODE_ISSETCARD:
            if stack:
                stack.append(_is_setcard(cd, stack.pop() & 0xFFFFFFFF))
        elif op == C.OPCODE_ALLOW_ALIASES:
            alias = True
        elif op == C.OPCODE_ALLOW_TOKENS:
            token = True
        else:
            stack.append(op)
    if len(stack) != 1 or stack[0] == 0:
        return False
    if cd.code in (C.CARD_MARINE_DOLPHIN, C.CARD_TWINKLE_MOSS):
        return True
    is_token = (cd.type & (C.TYPE_MONSTER | C.TYPE_TOKEN)) == (C.TYPE_MONSTER | C.TYPE_TOKEN)
    return (alias or not cd.alias) and (token or not is_token)


def declarable_codes(cards: Iterable[CardData], opcodes: Sequence[int]) -> list[int]:
    return [cd.code for cd in cards if is_declarable(cd, opcodes)]
