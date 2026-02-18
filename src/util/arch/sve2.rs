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

pub unsafe fn get_next_token1(data: &[u8; 16], tk: u8) -> usize {
    let mut idx: u64 = 16;
    let tk_wide = tk as u32; // 用 w 寄存器广播

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "ld1b   {{z0.b}}, p0/z, [{ptr}]",
        // 用 dup 将 32 位寄存器低 8 位广播到向量
        "dup    z1.b, {t:w}",

        "cmpeq  p1.b, p0/z, z0.b, z1.b",  // 找到匹配位置
        "b.none 1f",

        "brkb   p2.b, p0/z, p1.b",
        "cntp   {idx}, p0, p2.b",

        "1:",
        ptr = in(reg) data.as_ptr(),
        t   = in(reg) tk_wide,
        idx = inout(reg) idx,
        out("z0") _,
        out("z1") _,
        out("p0") _,
        out("p1") _,
        out("p2") _,
    );

    idx as usize
}

pub unsafe fn get_next_token2(data: &[u8; 16], tk1: u8, tk2: u8) -> usize {
    let mut idx: u64 = 16;
    // 将两个 token 填入 32 位寄存器（高低各复制一份，match 会扫描 z1 所有字节）
    let token = (tk1 as u32) | ((tk2 as u32) << 8) | ((tk1 as u32) << 16) | ((tk2 as u32) << 24);

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "ld1b   {{z0.b}}, p0/z, [{ptr}]",
        "mov    z1.s, {t:w}",

        "match  p1.b, p0/z, z0.b, z1.b", // 一条指令匹配两个 token
        "b.none 1f",

        "brkb   p2.b, p0/z, p1.b",
        "cntp   {idx}, p0, p2.b",

        "1:",
        ptr = in(reg) data.as_ptr(),
        t   = in(reg) token,
        idx = inout(reg) idx,
        out("z0") _, out("z1") _,
        out("p0") _, out("p1") _, out("p2") _,
    );

    idx as usize
}

// pub unsafe fn get_next_token2(data: &[u8; 16], tk1: u8, tk2: u8) -> usize {
//     let mut idx: u64 = 16;
//     let (t1, t2) = (tk1 as u32, tk2 as u32);

//     core::arch::asm!(
//         "ptrue  p0.b, vl16",
//         "ld1b   {{z0.b}}, p0/z, [{ptr}]",
//         "dup    z1.b, {t1:w}",
//         "dup    z2.b, {t2:w}",

//         // 两路并行比较，orr 合并
//         "cmpeq  p1.b, p0/z, z0.b, z1.b",
//         "cmpeq  p2.b, p0/z, z0.b, z2.b",
//         "orr    p3.b, p0/z, p1.b, p2.b",

//         "b.none 1f",

//         "brkb   p4.b, p0/z, p3.b",
//         "cntp   {idx}, p0, p4.b",

//         "1:",
//         ptr = in(reg) data.as_ptr(),
//         t1  = in(reg) t1,
//         t2  = in(reg) t2,
//         idx = inout(reg) idx,
//         out("z0") _,
//         out("z1") _,
//         out("z2") _,
//         out("p0") _,
//         out("p1") _,
//         out("p2") _,
//         out("p3") _,
//         out("p4") _,
//     );

//     idx as usize
// }

pub unsafe fn get_next_token3(data: &[u8; 16], tk1: u8, tk2: u8, tk3: u8) -> usize {
    let mut idx: u64 = 16;
    // 3 个 token 填入 32 位（第4字节复用 tk1，match 扫描所有字节，重复不影响正确性）
    let token = (tk1 as u32) | ((tk2 as u32) << 8) | ((tk3 as u32) << 16) | ((tk1 as u32) << 24); // 第4位复用，无副作用

    core::arch::asm!(
        "ptrue  p0.b, vl16",
        "ld1b   {{z0.b}}, p0/z, [{ptr}]",
        "mov    z1.s, {t:w}",

        "match  p1.b, p0/z, z0.b, z1.b", // 一条指令匹配三个 token
        "b.none 1f",

        "brkb   p2.b, p0/z, p1.b",
        "cntp   {idx}, p0, p2.b",

        "1:",
        ptr = in(reg) data.as_ptr(),
        t   = in(reg) token,
        idx = inout(reg) idx,
        out("z0") _, out("z1") _,
        out("p0") _, out("p1") _, out("p2") _,
    );

    idx as usize
}

#[test]
fn test_get_next_tokens() {
    let data = b"\t\r\n xxxxxxxxxxxx";
    assert_eq!(unsafe { get_next_token1(data, b' ') }, 3);
    assert_eq!(unsafe { get_next_token2(data, b' ', b'\t') }, 0);
    assert_eq!(unsafe { get_next_token3(data, b'\n', b'x', b'a') }, 2);
}
