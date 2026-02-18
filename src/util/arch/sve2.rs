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

use core::arch::asm;

/// 寻找 16 字节块中的第一个非数字字符的位置。
/// 如果全是数字，返回 16。
#[inline(always)]
pub unsafe fn skip_digit_sve2_first(data: &[u8; 16]) -> usize {
    let mut first: u64;
    let start: u32 = b'0' as u32;
    let range: u32 = 9;

    asm!(
        "ptrue  p0.b, vl16",
        "ld1b   z0.b, p0/z, [{ptr}]",

        "mov    z1.b, {start:w}",
        "mov    z2.b, {range:w}",

        // 识别非数字
        "sub    z0.b, z0.b, z1.b",
        "cmphi  p1.b, p0/z, z0.b, z2.b",

        // 寻找第一个非数字的位置
        "brkb   p2.b, p0/z, p1.b",
        "cntp   {first}, p0, p2.b",

        ptr = in(reg) data.as_ptr(),
        start = in(reg) start,
        range = in(reg) range,
        first = out(reg) first,

        out("z0") _, out("z1") _, out("z2") _,
        out("p0") _, out("p1") _, out("p2") _,
    );

    first as usize
}

/// 在 16 字节块中，忽略 `0..=skip_up_to` 的字符，寻找下一个非数字的位置。
/// 如果剩下的全是数字，返回 16。
#[inline(always)]
pub unsafe fn skip_digit_sve2_next(data: &[u8; 16], skip_up_to: usize) -> usize {
    let mut next_idx: u64;
    let start: u32 = b'0' as u32;
    let range: u32 = 9;
    let zero: u64 = 0;
    let skip_bound = skip_up_to as u64;

    asm!(
        "ptrue  p0.b, vl16",
        "ld1b   z0.b, p0/z, [{ptr}]",

        "mov    z1.b, {start:w}",
        "mov    z2.b, {range:w}",

        // 重新识别非数字
        "sub    z0.b, z0.b, z1.b",
        "cmphi  p1.b, p0/z, z0.b, z2.b",

        // 核心复用逻辑：生成忽略掩码 (lane_idx <= skip_bound)
        "whilels p3.b, {zero}, {skip_bound}",
        // 从 p1 中清除掉要忽略的前缀
        "bic    p1.b, p0/z, p1.b, p3.b",

        // 在剩余部分寻找第一个非数字
        "brkb   p2.b, p0/z, p1.b",
        "cntp   {next_idx}, p0, p2.b",

        ptr = in(reg) data.as_ptr(),
        start = in(reg) start,
        range = in(reg) range,
        zero = in(reg) zero,
        skip_bound = in(reg) skip_bound,
        next_idx = out(reg) next_idx,

        out("z0") _, out("z1") _, out("z2") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _,
    );

    next_idx as usize
}
