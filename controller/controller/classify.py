"""Mismatch classification. Pure functions, no DB access.

Flag representation (x86db): state["flag"] is the integer EFLAGS/RFLAGS value;
instruction_undefined_flags.flag holds names OF SF ZF AF CF PF, mapped to the
architectural EFLAGS bit positions below. All other keys are compared exactly, except
the destination of SHLD/SHRD when the count exceeds the operand size (see undefined_for).
"""

FLAG_BITS = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "OF": 11}
ALL_FLAGS = tuple(FLAG_BITS)

UNDEF_FLAGS_ONLY = "undef_flags_only"
DEFINED_STATE = "defined_state"
EXCEPTION_MISMATCH = "exception_mismatch"
CRASH = "crash"
SKIPPED = "skipped"
CLASSES = (UNDEF_FLAGS_ONLY, DEFINED_STATE, EXCEPTION_MISMATCH, CRASH)

# Undefined for every operand (SDM "Flags Affected") but missing from instruction_undefined_flags.
EXTRA_UNDEFINED = {"DIV": ALL_FLAGS, "IDIV": ALL_FLAGS}

SHIFTS = {"SHL", "SAL", "SHR", "SAR"}
ROTATES = {"ROL", "ROR", "RCL", "RCR"}
DOUBLE_SHIFTS = {"SHLD", "SHRD"}

# SQL for the initial state a query must pass to classify(): only fetched for the instructions
# whose undefined flags depend on it (tc = test_cases, tr = test_results).
INITIAL_STATE_SQL = ("CASE WHEN upper(split_part(tc.instruction, ' ', 1)) = ANY(%s) "
                     "THEN tc.initial_states -> tr.state_index END")
INITIAL_STATE_ARGS = (sorted(SHIFTS | ROTATES | DOUBLE_SHIFTS),)

_SIZES = {"byte": 8, "word": 16, "dword": 32, "qword": 64}
_REG64 = {"a": "rax", "b": "rbx", "c": "rcx", "d": "rdx"}


def undefined_mask(flags) -> int:
    m = 0
    for f in flags or ():
        m |= 1 << FLAG_BITS[f.strip().upper()]
    return m


def _operand(op: str) -> tuple[int, str] | None:
    """(size in bits, state key) of a destination operand; None when it is not understood."""
    op = op.strip()
    if "ptr" in op:
        size = _SIZES.get(op.split()[0])
        return (size, "mem0_value") if size else None
    if op in ("al", "bl", "cl", "dl"):
        return 8, _REG64[op[0]]
    if op in ("ax", "bx", "cx", "dx"):
        return 16, _REG64[op[0]]
    if len(op) == 3 and op[0] in "er" and op[1] in _REG64 and op[2] == "x":
        return (32 if op[0] == "e" else 64), _REG64[op[1]]
    return None


def undefined_for(instruction: str | None, initial_state: dict | None, flags) -> tuple[int, dict]:
    """(undefined flag mask, {state key: mask of undefined bits}) for one state.

    For shifts and rotates the SDM makes flags undefined only for some masked counts: a count
    of 0 changes nothing, a count of 1 defines OF, and SHLD/SHRD with a count above the operand
    size leave the result and every flag undefined. Other instructions use their listed flags,
    plus EXTRA_UNDEFINED. Without an instruction or initial state, the listed flags are used."""
    if not instruction:
        return undefined_mask(flags), {}
    mnemonic, _, rest = instruction.strip().partition(" ")
    mnemonic = mnemonic.upper()
    listed = undefined_mask(flags) | undefined_mask(EXTRA_UNDEFINED.get(mnemonic))
    if mnemonic not in SHIFTS | ROTATES | DOUBLE_SHIFTS or initial_state is None:
        return listed, {}
    ops = [o.strip() for o in rest.split(",")]
    dest = _operand(ops[0])
    if dest is None or len(ops) < 2:
        return listed, {}
    size, key = dest
    try:
        raw = int(initial_state.get("rcx", 0)) & 0xFF if ops[-1] == "cl" else int(ops[-1], 0) & 0xFF
    except ValueError:
        return listed, {}
    count = raw & (0x3F if size == 64 else 0x1F)
    if count == 0:
        return 0, {}
    if mnemonic in ROTATES:
        return (0 if count == 1 else undefined_mask(["OF"])), {}
    if mnemonic in DOUBLE_SHIFTS and count > size:
        return undefined_mask(ALL_FLAGS), {key: (1 << size) - 1}
    undef = ["AF"] if count == 1 else ["AF", "OF"]
    if mnemonic in ("SHL", "SAL", "SHR") and count >= size:
        undef.append("CF")
    return undefined_mask(undef), {}


def diff_keys(expected: dict | None, got: dict | None) -> list[str]:
    expected = expected or {}
    got = got or {}
    return sorted(k for k in set(expected) | set(got) if expected.get(k, _MISSING) != got.get(k, _MISSING))


_MISSING = object()


def uncaptured_in_fxsave(key: str) -> bool:
    """Keys the kernel cannot capture in FXSAVE mode: ymmN, zmmN, kN, opmask*."""
    for p in ("ymm", "zmm", "k"):
        if key.startswith(p) and key[len(p):].isdigit():
            return True
    return key.startswith("opmask")


def strip_uncaptured(state: dict | None) -> dict | None:
    if state is None:
        return None
    return {k: v for k, v in state.items() if not uncaptured_in_fxsave(k)}


def _within(e, g, mask: int) -> bool:
    try:
        return (int(e) ^ int(g)) & ~mask == 0
    except (TypeError, ValueError):
        return False


def classify(status, expected_state, expected_exc, got_state, got_exc, undefined_flags,
             save_mode=None, instruction=None, initial_state=None):
    """Return (class, differing_keys). In fxsave mode the wide vector/opmask keys
    were not captured and are excluded from the comparison (nothing else is).
    With the instruction text and initial state, the undefined flags (and bits of the
    destination) are those of that state, see undefined_for."""
    if save_mode == "fxsave":
        expected_state = strip_uncaptured(expected_state)
        got_state = strip_uncaptured(got_state)
    if status == "crash":
        return CRASH, []
    if status == "skipped":
        return SKIPPED, []
    if (expected_exc or None) != (got_exc or None):
        return EXCEPTION_MISMATCH, []
    keys = diff_keys(expected_state, got_state)
    if "flag" in keys:
        flag_mask, free = undefined_for(instruction, initial_state, undefined_flags)
        if all(_within(expected_state.get(k), got_state.get(k), flag_mask if k == "flag" else free.get(k, 0))
               for k in keys):
            return UNDEF_FLAGS_ONLY, keys
    return DEFINED_STATE, keys
