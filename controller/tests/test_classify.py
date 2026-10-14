from controller.classify import (DEFINED_STATE, UNDEF_FLAGS_ONLY, classify, undefined_for,
                                 undefined_mask)

CF, PF, AF, ZF, SF, OF = (undefined_mask([f]) for f in ("CF", "PF", "AF", "ZF", "SF", "OF"))


def run(instruction, init, expected, got, flags=()):
    return classify("mismatch", expected, None, got, None, list(flags), "xsave", instruction, init)


def test_without_instruction_uses_listed_flags():
    assert undefined_for(None, None, ["AF", "OF"]) == (AF | OF, {})
    assert run(None, None, {"flag": 0}, {"flag": OF}, ["OF"]) == (UNDEF_FLAGS_ONLY, ["flag"])


def test_div_idiv_all_flags_undefined():
    for ins in ("div rbx", "idiv byte ptr [r8 + 0x0]"):
        assert undefined_for(ins, {}, [])[0] == CF | PF | AF | ZF | SF | OF
        assert run(ins, {}, {"flag": 2, "rax": 1}, {"flag": 2 | AF, "rax": 1}) == (UNDEF_FLAGS_ONLY, ["flag"])
        assert run(ins, {}, {"flag": 2, "rax": 1}, {"flag": 2 | AF, "rax": 2})[0] == DEFINED_STATE


def test_shift_count_zero_defines_everything():
    # imm8 -0x80 = 0x80, masked to 5 bits = 0
    assert undefined_for("shl al, -0x80", {}, ["AF", "OF"]) == (0, {})
    assert undefined_for("sar eax, cl", {"rcx": 0x20}, ["AF", "OF"]) == (0, {})
    assert run("shr ax, cl", {"rcx": 0}, {"flag": 2}, {"flag": 2 | AF}, ["AF", "OF"])[0] == DEFINED_STATE


def test_shift_count_one_defines_of():
    assert undefined_for("shl rax, 0x1", {}, ["AF", "OF"])[0] == AF
    assert run("sal al, 0x1", {}, {"flag": 2}, {"flag": 2 | OF}, ["AF", "OF"])[0] == DEFINED_STATE
    assert run("sal al, 0x1", {}, {"flag": 2}, {"flag": 2 | AF}, ["AF", "OF"])[0] == UNDEF_FLAGS_ONLY


def test_shift_count_above_one_and_size():
    assert undefined_for("sar al, 0x2", {}, [])[0] == AF | OF
    # 8-bit operand, count 30: CF undefined for SHL/SHR, not for SAR
    assert undefined_for("shl byte ptr [r8 + 0x0], 0x7e", {}, [])[0] == AF | OF | CF
    assert undefined_for("shr al, cl", {"rcx": 0x108}, [])[0] == AF | OF | CF  # cl = 8
    assert undefined_for("sar al, 0x7e", {}, [])[0] == AF | OF
    # 64-bit: 6-bit count, 0x7e -> 62 < 64
    assert undefined_for("shl rax, 0x7e", {}, [])[0] == AF | OF
    assert undefined_for("shl eax, -0x1", {}, [])[0] == AF | OF  # 31 < 32


def test_rotates():
    assert undefined_for("rol al, 0x1", {}, ["OF"])[0] == 0
    assert undefined_for("rcr rax, -0x80", {}, ["OF"])[0] == 0
    assert undefined_for("ror word ptr [r8 + 0x0], cl", {"rcx": 3}, ["OF"])[0] == OF
    assert run("rol al, 0x1", {}, {"flag": 2}, {"flag": 2 | OF}, ["OF"])[0] == DEFINED_STATE
    assert run("rol al, 0x2", {}, {"flag": 2}, {"flag": 2 | OF}, ["OF"])[0] == UNDEF_FLAGS_ONLY


def test_double_shift_count_above_size():
    # 16-bit, count 31 > 16: all flags and the low 16 bits of the destination undefined
    mask, free = undefined_for("shld ax, bx, -0x1", {}, ["AF", "OF"])
    assert mask == CF | PF | AF | ZF | SF | OF and free == {"rax": 0xFFFF}
    exp, init = {"flag": 2, "rax": 0x1111_0000}, {}
    assert run("shld ax, bx, -0x1", init, exp, {"flag": 2 | CF | ZF, "rax": 0x1111_1234}) == (
        UNDEF_FLAGS_ONLY, ["flag", "rax"])
    # bits above the operand are preserved, so a difference there is still defined
    assert run("shld ax, bx, -0x1", init, exp, {"flag": 2, "rax": 0x2222_0000})[0] == DEFINED_STATE
    _, free = undefined_for("shrd word ptr [r8 + 0x20], bx, cl", {"rcx": 17}, [])
    assert free == {"mem0_value": 0xFFFF}
    # count 16 is not above the size, 32-bit forms never are
    assert undefined_for("shrd ax, bx, cl", {"rcx": 16}, [])[0] == AF | OF
    assert undefined_for("shld eax, ebx, -0x1", {}, [])[0] == AF | OF


def test_double_shift_one_and_zero():
    assert undefined_for("shrd rax, rbx, 0x1", {}, ["AF", "OF"])[0] == AF
    assert undefined_for("shld rax, rbx, 0x0", {}, ["AF", "OF"])[0] == 0


def test_unparsed_operand_falls_back_to_listed():
    assert undefined_for("shl r9b, 0x2", {}, ["AF", "OF"]) == (AF | OF, {})


def test_flag_only_rule_unchanged_for_other_keys():
    assert run("mul rbx", {}, {"flag": 2, "rax": 1}, {"flag": 2 | SF, "rax": 2}, ["SF"])[0] == DEFINED_STATE
    assert run("mul rbx", {}, {"flag": 2}, {"flag": 2 | CF}, ["SF"])[0] == DEFINED_STATE
