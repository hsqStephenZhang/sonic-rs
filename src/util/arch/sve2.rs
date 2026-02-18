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

/// SVE2 implementation: Returns the index of the first non-space char (0-15).
/// Returns 16 if all characters are spaces.
#[inline(always)]
pub unsafe fn get_nonspace_index(data: &[u8; 16]) -> usize {
    let mut idx: u64 = 16; // Default to 16 (Not Found)
                           // 0x09 (Tab), 0x0A (LF), 0x0D (CR), 0x20 (Space)
    let tokens: u32 = 0x090a0d20;

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "ld1b   {{z0.b}}, p0/z, [{ptr}]",
        "mov    z1.s, {t:w}",

        // 1. Identify non-space characters
        // NMATCH sets the Z flag if NO non-spaces are found (all whitespace)
        "nmatch p1.b, p0/z, z0.b, z1.b",

        // 2. Fast Path: Branch if NO non-space characters were found.
        // b.none checks the Z flag set by nmatch.
        // If Z=1 (all spaces), we skip the calculation and keep idx=16.
        "b.none 1f",

        // 3. Slow Path (Found something): Calculate the exact index
        "brkb   p2.b, p0/z, p1.b", // Mask bits *after* the first match
        "cntp   {idx}, p0, p2.b",  // Count leading matches

        "1:",
        ptr = in(reg) data.as_ptr(),
        t = in(reg) tokens,
        idx = inout(reg) idx,
        out("z0") _, out("z1") _,
        out("p0") _, out("p1") _, out("p2") _,
    );

    idx as usize
}

#[inline(always)]
pub unsafe fn skip_digit_sve2_pair(data: &[u8; 16]) -> (usize, usize) {
    let mut first: u64;
    let mut second: u64;

    let start: u32 = b'0' as u32;
    let range: u32 = 9;

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "ld1b   z0.b, p0/z, [{ptr}]",

        "mov    z1.b, {start:w}",
        "mov    z2.b, {range:w}",

        // 1. identify all non-digits
        "sub    z0.b, z0.b, z1.b",
        "cmphi  p1.b, p0/z, z0.b, z2.b",

        // 2. first non-digit
        "brkb   p2.b, p0/z, p1.b",
        "cntp   {first}, p0, p2.b",

        // 3. mask out first non-digit
        "brka   p3.b, p0/z, p1.b",
        "bic    p4.b, p0/z, p1.b, p3.b",

        // 4. second non-digit
        "brkb   p5.b, p0/z, p4.b",
        "cntp   {second}, p0, p5.b",

        ptr = in(reg) data.as_ptr(),
        start = in(reg) start,
        range = in(reg) range,
        first = out(reg) first,
        second = out(reg) second,

        out("z0") _,
        out("z1") _,
        out("z2") _,
        out("p0") _,
        out("p1") _,
        out("p2") _,
        out("p3") _,
        out("p4") _,
        out("p5") _,
    );

    (first as usize, second as usize)
}

#[test]
fn test_skip_digit_sve2_pair() {
    let mut s = [0u8; 16];
    let dst = b"12.34e5";
    s[..dst.len()].copy_from_slice(&dst[..]);
    let (idx1, idx2) = unsafe { skip_digit_sve2_pair(&s) };
    assert_eq!(idx1, 2); // '.' at index 2
    assert_eq!(idx2, 5); // 'e' at index 5
}
