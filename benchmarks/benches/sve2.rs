#![allow(warnings)]

use std::{fs::read_dir, hint::black_box};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sonic_rs::{Read, prelude::Reader};

#[inline(always)]
fn is_whitespace(ch: u8) -> bool {
    const SPACE_MASK: u64 = (1u64 << b' ') | (1u64 << b'\r') | (1u64 << b'\n') | (1u64 << b'\t');
    1u64.checked_shl(ch as u32)
        .is_some_and(|v| v & SPACE_MASK != 0)
}

struct NeonSpaceSkipper {
    nospace_bits: u64,
    nospace_start: isize,
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
        pub unsafe fn get_nonspace_bits(data: &[u8; 64]) -> u64 {
            use std::arch::aarch64::*;

            #[inline(always)]
            unsafe fn chunk_nonspace_bits(input: uint8x16_t) -> uint8x16_t {
                const LOW_TAB: uint8x16_t = unsafe {
                    std::mem::transmute([16u8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0])
                };
                const HIGH_TAB: uint8x16_t = unsafe {
                    std::mem::transmute([8u8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0])
                };

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
                reader.set_index(self.nospace_start as usize + 64);
            }
        }

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

    #[inline(always)]
    pub fn skip_all_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) {
        while self.skip_space(reader).is_some() {}
    }
}

// ==========================================
// SVE2 版本
// ==========================================
struct SveSpaceSkipper;

impl SveSpaceSkipper {
    pub fn new() -> Self {
        Self
    }

    #[inline(always)]
    pub unsafe fn skip_space_sve2<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        #[inline(always)]
        unsafe fn get_nonspace_bits(data: &[u8; 16]) -> u64 {
            let mut index: u64;
            let tokens: u32 = 0x090a0d20;

            core::arch::asm!(
                "ptrue  p0.b, vl16",
                "ld1b   {{z0.b}}, p0/z, [{ptr}]",
                "mov    z1.s, {t:w}",
                "nmatch p1.b, p0/z, z0.b, z1.b",
                "brkb   p1.b, p0/z, p1.b",
                "cntp   {idx}, p0, p1.b",
                ptr = in(reg) data.as_ptr(),
                t = in(reg) tokens,
                idx = out(reg) index,
                out("z0") _, out("z1") _,
                out("p0") _, out("p1") _,
            );

            if index < 16 { 1u64 << index } else { 0 }
        }

        while let Some(chunk) = reader.peek_n(16) {
            let chunk = unsafe { &*(chunk.as_ptr() as *const [_; 16]) };
            let bitmap = unsafe { get_nonspace_bits(chunk) };
            if bitmap != 0 {
                let cnt = bitmap.trailing_zeros() as usize;
                let ch = chunk[cnt];
                reader.eat(cnt + 1);
                return Some(ch);
            }
            reader.eat(16)
        }

        while let Some(ch) = reader.next() {
            if !is_whitespace(ch) {
                return Some(ch);
            }
        }
        None
    }

    #[inline(always)]
    pub fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        unsafe { self.skip_space_sve2(reader) }
    }

    #[inline(always)]
    pub fn skip_all_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) {
        while self.skip_space(reader).is_some() {}
    }
}

struct SveSpaceSkipperWithCache {
    nospace_bits: u64,
    nospace_start: isize,
}

impl SveSpaceSkipperWithCache {
    pub fn new() -> Self {
        Self {
            nospace_bits: 0,
            nospace_start: -128,
        }
    }

    #[inline(always)]
    pub unsafe fn skip_space_sve2<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        #[inline(always)]
        unsafe fn get_nonspace_bits(data: &[u8; 16]) -> u64 {
            let mut pred_buf = [0u8; 32];
            let tokens: u32 = 0x090a0d20;

            core::arch::asm!(
                "ptrue  p0.b, vl16",
                "ld1b   {{z0.b}}, p0/z, [{ptr}]",
                "mov    z1.s, {t:w}",
                "nmatch p1.b, p0/z, z0.b, z1.b",
                "str    p1, [{out_ptr}]",
                ptr = in(reg) data.as_ptr(),
                t = in(reg) tokens,
                out_ptr = in(reg) pred_buf.as_mut_ptr(),
                out("z0") _, out("z1") _,
                out("p0") _, out("p1") _,
            );

            u16::from_le_bytes([pred_buf[0], pred_buf[1]]) as u64
        }

        let nospace_offset = (reader.index() as isize) - self.nospace_start;
        if nospace_offset < 16 {
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
                reader.set_index(self.nospace_start as usize + 16);
            }
        }

        while let Some(chunk) = reader.peek_n(16) {
            let chunk = unsafe { &*(chunk.as_ptr() as *const [_; 16]) };
            let bitmap = unsafe { get_nonspace_bits(chunk) };
            if bitmap != 0 {
                self.nospace_bits = bitmap;
                self.nospace_start = reader.index() as isize;
                let cnt = bitmap.trailing_zeros() as usize;
                let ch = chunk[cnt];
                reader.eat(cnt + 1);
                return Some(ch);
            }
            reader.eat(16)
        }

        while let Some(ch) = reader.next() {
            if !is_whitespace(ch) {
                return Some(ch);
            }
        }
        None
    }

    #[inline(always)]
    pub fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
        unsafe { self.skip_space_sve2(reader) }
    }

    #[inline(always)]
    pub fn skip_all_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) {
        while self.skip_space(reader).is_some() {}
    }
}

fn bench_space_skipper(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceSkipper_RealData");

    let testdata_dir = std::env::var("TESTDIR").unwrap();
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();

    match std::fs::read_dir(&testdata_dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    let file_name = path.file_stem().unwrap().to_string_lossy().to_string();
                    let content = std::fs::read(&path)
                        .unwrap_or_else(|_| panic!("Failed to read {:?}", path));
                    files.push((file_name, content));
                }
            }
        }
        Err(e) => {
            eprintln!(
                "⚠️ Warning: Failed to read directory '{}': {}. Skipping file-based benchmarks.",
                testdata_dir, e
            );
            return;
        }
    }

    if files.is_empty() {
        eprintln!("⚠️ Warning: No .json files found in '{}'.", &testdata_dir);
        return;
    }

    for (name, payload) in files {
        // NEON
        group.bench_with_input(BenchmarkId::new("NEON", &name), &payload, |b, p| {
            b.iter(|| {
                let mut reader = Read::from(p.as_slice());
                let mut skipper = NeonSpaceSkipper::new();
                black_box(skipper.skip_all_space(&mut reader))
            });
        });

        // SVE2
        group.bench_with_input(BenchmarkId::new("SVE2", &name), &payload, |b, p| {
            b.iter(|| {
                let mut reader = Read::from(p.as_slice());
                let mut skipper = SveSpaceSkipper::new();
                black_box(skipper.skip_all_space(&mut reader))
            });
        });

        // SVE2 - cached
        group.bench_with_input(BenchmarkId::new("SVE2-bitmask", &name), &payload, |b, p| {
            b.iter(|| {
                let mut reader = Read::from(p.as_slice());
                let mut skipper = SveSpaceSkipperWithCache::new();
                black_box(skipper.skip_all_space(&mut reader))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_space_skipper);
criterion_main!(benches);
