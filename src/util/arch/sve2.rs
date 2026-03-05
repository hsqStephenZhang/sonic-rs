// Not use PMULL instructions, but it is apparently slow.
// This is copied from simdjson.
pub unsafe fn prefix_xor(bitmask: u64) -> u64 {
    let mut bitmask = bitmask;
    bitmask ^= bitmask << 1;
    bitmask ^= bitmask << 2;
    bitmask ^= bitmask << 4;
    bitmask ^= bitmask << 8;
    bitmask ^= bitmask << 16;
    bitmask ^= bitmask << 32;
    bitmask
}

pub unsafe fn get_nonspace_bits(data: &[u8; 64]) -> u64 {
    let mut pred_buf = [0u64; 8];
    let tokens: u32 = 0x090a0d20;

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "mov    z4.s, {t:w}",

        "mov    x10, #16",
        "mov    x11, #32",
        "mov    x12, #48",

        "ld1b   {{z0.b}}, p0/z, [{ptr}]",
        "ld1b   {{z1.b}}, p0/z, [{ptr}, x10]",
        "ld1b   {{z2.b}}, p0/z, [{ptr}, x11]",
        "ld1b   {{z3.b}}, p0/z, [{ptr}, x12]",

        "nmatch p1.b, p0/z, z0.b, z4.b",
        "nmatch p2.b, p0/z, z1.b, z4.b",
        "nmatch p3.b, p0/z, z2.b, z4.b",
        "nmatch p4.b, p0/z, z3.b, z4.b",

        "add    x10, {out_ptr}, #2",
        "add    x11, {out_ptr}, #4",
        "add    x12, {out_ptr}, #6",

        "str    p1, [{out_ptr}]",
        "str    p2, [x10]",
        "str    p3, [x11]",
        "str    p4, [x12]",

        ptr = in(reg) data.as_ptr(),
        t = in(reg) tokens,
        out_ptr = in(reg) pred_buf.as_mut_ptr(),

        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
        out("x10") _, out("x11") _, out("x12") _
    );

    pred_buf[0]
}