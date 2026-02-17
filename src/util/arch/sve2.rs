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
    let mut first: usize;
    let mut second: usize;

    let start: u32 = b'0' as u32;
    let range: u32 = 9;

    // 所有的逻辑都在这里完成，避免将 predicate 导出到 Rust
    core::arch::asm!(
        "ptrue  p0.b, vl16",            // 激活前16个 lane
        "ld1b   {z_data}.b, p0/z, [{ptr}]",

        "mov    z1.b, {start:w}",
        "mov    z2.b, {range:w}",

        // 1. 识别所有非数字 (Check digits)
        "sub    z0.b, {z_data}.b, z1.b",
        "cmphi  p1.b, p0/z, z0.b, z2.b", // p1 中 True 的位置就是非数字

        // 2. 找第一个非数字 (Find first)
        "brkb   p2.b, p0/z, p1.b",       // p2 = True 直到遇到第一个 p1 的 True
        "cntp   {first}, p0, p2.b",      // first = p2 中 True 的数量 (即 index)

        // 3. 准备找第二个 (Prepare for second)
        // brka: Break After. p3 会包含从头开始直到(包含) p1 中第一个 True
        "brka   p3.b, p0/z, p1.b",       
        
        // bic: Bitwise Clear. 从 p1 中清除掉 p3 标记的位
        // 也就是把第一个非数字(以及之前的无效位)从掩码中抹去
        "bic    p4.b, p0/z, p1.b, p3.b", 

        // 4. 找第二个非数字 (Find second)
        // 注意：brkb 会把第一个 Active 之前的位全置为 True
        // 如果 p4 是空的(没有第二个非数字)，brkb 会全置 True (受 p0 限制)，cntp 结果为 16
        // 如果 p4 在 index 8 是 True，brkb 会把 0..7 置 True，cntp 结果为 8
        "brkb   p5.b, p0/z, p4.b",
        "cntp   {second}, p0, p5.b",

        ptr = in(reg) data.as_ptr(),
        start = in(reg) start,
        range = in(reg) range,
        first = out(reg) first,
        second = out(reg) second,
        z_data = out("z0"),
        out("z1") _, out("z2") _,
        out("p0") _, out("p1") _, out("p2") _, 
        out("p3") _, out("p4") _, out("p5") _,
    );

    (first, second)
}
