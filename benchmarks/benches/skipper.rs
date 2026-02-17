struct NeonSpaceSkipper {
    nospace_bits: u64,    // SIMD marked nospace bitmap
    nospace_start: isize, // the start position of nospace_bits
}

impl NeonSpaceSkipper {
    pub fn new() -> Self {
        Self {
            nospace_bits: 0,
            nospace_start: -128,
        }
    }

    #[inline(always)]
    pub fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        // fast path 2: reuse the bitmap for short key or numbers
        let nospace_offset = (reader.index() as isize) - self.nospace_start;
        if nospace_offset < 64 {
            let bitmap = {
                let mask = !((1 << nospace_offset) - 1);
                self.nospace_bits & mask
            };
            if bitmap != 0 {
                let cnt = bitmap.trailing_zeros() as usize;
                let ch = reader.at(self.nospace_start as usize + cnt);
                reader.set_index(self.nospace_start as usize + cnt + 1);

                return Some(ch);
            } else {
                // we can still fast skip the marked space in here.
                reader.set_index(self.nospace_start as usize + 64);
            }
        }

        // then we use simd to accelerate skipping space
        while let Some(chunk) = reader.peek_n(64) {
            let chunk = unsafe { &*(chunk.as_ptr() as *const [_; 64]) };
            let bitmap = unsafe { get_nonspace_bits(chunk) };
            if bitmap != 0 {
                self.nospace_bits = bitmap;
                self.nospace_start = reader.index() as isize;
                let cnt = bitmap.trailing_zeros() as usize;
                let ch = chunk[cnt];
                reader.eat(cnt + 1);

                return Some(ch);
            }
            reader.eat(64)
        }

        while let Some(ch) = reader.next() {
            if !is_whitespace(ch) {
                return Some(ch);
            }
        }
        None
    }
}


// We compute whitespace and op separately. If the code later only use one or the
// other, given the fact that all functions are aggressively inlined, we can
// hope that useless computations will be omitted. This is namely case when
// minifying (we only need whitespace). *However* if we only need spaces,
// it is likely that we will still compute 'v' above with two lookup_16: one
// could do it a bit cheaper. This is in contrast with the x64 implementations
// where we can, efficiently, do the white space and structural matching
// separately. One reason for this difference is that on ARM NEON, the table
// lookups either zero or leave unchanged the characters exceeding 0xF whereas
// on x64, the equivalent instruction (pshufb) automatically applies a mask,
// ignoring the 4 most significant bits. Thus the x64 implementation is
// optimized differently. This being said, if you use this code strictly
// just for minification (or just to identify the structural characters),
// there is a small untaken optimization opportunity here. We deliberately
// do not pick it up.
#[inline(always)]
pub unsafe fn get_nonspace_bits(data: &[u8; 64]) -> u64 {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn chunk_nonspace_bits(input: uint8x16_t) -> uint8x16_t {
        const LOW_TAB: uint8x16_t =
            unsafe { std::mem::transmute([16u8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0]) };

        const HIGH_TAB: uint8x16_t =
            unsafe { std::mem::transmute([8u8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0]) };

        let white_mask = vmovq_n_u8(0x18);
        let lo4 = vandq_u8(input, vmovq_n_u8(0xf));
        let hi4 = vshrq_n_u8(input, 4);

        let lo4_sf = vqtbl1q_u8(LOW_TAB, lo4);
        let hi4_sf = vqtbl1q_u8(HIGH_TAB, hi4);

        let v = vandq_u8(lo4_sf, hi4_sf);

        vtstq_u8(v, white_mask)
    }

    !sonic_simd::neon::to_bitmask64(
        chunk_nonspace_bits(vld1q_u8(data.as_ptr())),
        chunk_nonspace_bits(vld1q_u8(data.as_ptr().offset(16))),
        chunk_nonspace_bits(vld1q_u8(data.as_ptr().offset(32))),
        chunk_nonspace_bits(vld1q_u8(data.as_ptr().offset(48))),
    )
}


struct SVESpaceSkipper;

impl SVESpaceSkipper {
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        // then we use simd to accelerate skipping space
        while let Some(chunk) = reader.peek_n(16) {
            let chunk = unsafe { &*(chunk.as_ptr() as *const [_; 16]) };
            let cnt = unsafe { get_nonspace_index(chunk) };

            if cnt < 16 {
                let ch = chunk[cnt];
                reader.eat(cnt + 1); // Skip spaces + return char
                return Some(ch);
            }
            reader.eat(16)
        }

        while let Some(ch) = reader.next() {
            if !is_whitespace(ch) {
                //
                return Some(ch);
            }
        }
        None
    }
}



/// SVE2 implementation: Returns the index of the first non-space char (0-15).
/// Returns 16 if all characters are spaces.
#[cfg(target_feature = "sve2")]
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

#[cfg(not(target_feature = "sve2"))]
#[inline(always)]
pub unsafe fn get_nonspace_index(data: &[u8; 16]) -> usize {
    // Fallback scalar scan when SVE2 is unavailable
    for i in 0..16 {
        if !is_whitespace(data[i]) {
            return i;
        }
    }
    16
}

#[inline(always)]
fn is_whitespace(ch: u8) -> bool {
    const SPACE_MASK: u64 = (1u64 << b' ') | (1u64 << b'\r') | (1u64 << b'\n') | (1u64 << b'\t');
    match 1u64.checked_shl(ch as u32) {
        Some(v) => (v & SPACE_MASK) != 0,
        None => false,
    }
}

use criterion::{black_box, Criterion, BenchmarkId, criterion_group, criterion_main};
use sonic_rs::{Read, Reader};

fn make_random_with_n_leading_spaces(n_spaces: usize, total_len: usize) -> Vec<u8> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let total_len = total_len.max(n_spaces + 1);
    let mut rng = StdRng::from_seed([42u8; 32]);
    let mut buf = vec![b'A'; total_len];
    let whites: [u8; 4] = [b' ', b'\t', b'\n', b'\r'];
    for i in 0..n_spaces {
        let w = whites[rng.r#gen::<usize>() % whites.len()];
        buf[i] = w;
    }
    // Ensure first non-space right after the run
    buf[n_spaces] = b'X';
    // Fill the rest with random non-whitespace ASCII
    for i in (n_spaces + 1)..total_len {
        let mut c = rng.gen_range(33u8..=126u8);
        // Avoid whitespace
        while is_whitespace(c) {
            c = rng.gen_range(33u8..=126u8);
        }
        buf[i] = c;
    }
    buf
}

fn bench_skipper(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_skipper");
    for &n in &[0usize, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let data = make_random_with_n_leading_spaces(n, n + 64);

        group.bench_with_input(BenchmarkId::new("neon", n), &data, |b, data| {
            let mut reader = Read::from(data.as_slice());
            let mut skipper = NeonSpaceSkipper::new();
            b.iter(|| {
                reader.set_index(0);
                let ch = skipper.skip_space(&mut reader);
                black_box(ch)
            });
        });

        group.bench_with_input(BenchmarkId::new("sve2", n), &data, |b, data| {
            let mut reader = Read::from(data.as_slice());
            let mut skipper = SVESpaceSkipper::new();
            b.iter(|| {
                reader.set_index(0);
                let ch = skipper.skip_space(&mut reader);
                black_box(ch)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_skipper);
criterion_main!(benches);


