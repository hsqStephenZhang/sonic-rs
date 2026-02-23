use core::arch::aarch64::*;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::RngExt;

/// we use unzip to extract even and odd indexed digits, then we can do multiplication and addition
/// in parallel. for [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3, 2, 1, 0, 0]
/// it will be split into even = [1, 3, 5, 7, 9, 4, 2, 0, ...] and odd = [2, 4, 6, 8, 5, 3, 1, 0,
/// ...] then we can compute 1*10 + 2, 3*10 + 4, 5*10 + 6, 7*10 + 8, 9*10 + 5, 4*10 + 3, 2*10 + 1,
/// 0*10 + 0 in parallel. so we get [12, 34, 56, 78, 95, 43, 21, 0]
macro_rules! packadd_1 {
    ($v:expr) => {
        unsafe {
            let even = vuzp1q_u8($v, $v);
            let odd = vuzp2q_u8($v, $v);
            vaddw_u8(vmull_u8(vget_low_u8(even), vdup_n_u8(10)), vget_low_u8(odd))
        }
    };
}

/// should be called after packadd_1
/// for [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3, 2, 1, 0, 0]
/// we get [12, 34, 56, 78, 95, 43, 21, 0] after packadd_1
/// here, it will be split into even = [12, 56, 95, 21] and odd = [34, 78, 43, 0]
/// then we can compute 12*100 + 34, 56*100 + 78, 95*100 + 43, 21*100 + 0 in parallel.
/// so we get [1234, 5678, 9543, 2100]
macro_rules! packadd_2 {
    ($v:expr) => {
        unsafe {
            let even = vuzp1q_u16($v, $v);
            let odd = vuzp2q_u16($v, $v);
            vaddw_u16(vmull_n_u16(vget_low_u16(even), 100), vget_low_u16(odd))
        }
    };
}

/// should be called after packadd_2, it will compute 4 digits in parallel
/// for [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3, 2, 1, 0, 0]
/// we get [1234, 5678, 9543, 2100] after packadd_2
/// here, it will be split into even = [1234, 9543] and odd = [5678, 2100]
/// then we can compute 1234*10000 + 5678 and 9543*10000 + 2100 in parallel,
/// so we get [12345678, 95432100]
macro_rules! packadd_4 {
    ($v:expr) => {
        unsafe {
            let even = vuzp1q_u32($v, $v);
            let odd = vuzp2q_u32($v, $v);
            vaddw_u32(vmull_n_u32(vget_low_u32(even), 10000), vget_low_u32(odd))
        }
    };
}

macro_rules! simd_add_5_8 {
    ($v:ident, $count:literal) => {{
        let shifted = vextq_u8::<$count>(vdupq_n_u8(0), $v);
        let p1 = packadd_1!(shifted);
        let p2 = packadd_2!(p1);
        (vgetq_lane_u32::<2>(p2) as u64) * 10000 + (vgetq_lane_u32::<3>(p2) as u64)
    }};
}

macro_rules! simd_add_8 {
    ($v:ident) => {{
        let p1 = packadd_1!($v);
        let p2 = packadd_2!(p1);
        packadd_4!(p2)
    }};
}

/// how it works:
/// for "123456789", we have [1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
/// calling `vextq_u8::<N>` will keep N bytes of the original vector and align them to the right,
/// and fill the left with zeros. so we get
/// shift = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (16 - N zeros)
/// then its aligned to the right, we could call simd_add_8 to get [12345678, 9, ..]
/// and extract the first and second lane to get the final result.
macro_rules! simd_add_9_15 {
    ($v:ident, $count:literal) => {{
        let shifted = vextq_u8::<$count>(vdupq_n_u8(0), $v);
        let p4 = simd_add_8!(shifted);
        vgetq_lane_u64::<0>(p4) * 100000000 + vgetq_lane_u64::<1>(p4)
    }};
}

#[inline(always)]
pub unsafe fn simd_str2int_vertical_vext(c: &[u8], need: usize) -> (u64, usize) {
    debug_assert!(need <= 16);

    let data = vld1q_u8(c.as_ptr());
    let zero_char = vdupq_n_u8(b'0');

    let digits = vsubq_u8(data, zero_char);
    let gt_nine = vcgtq_u8(digits, vdupq_n_u8(9));

    let mask16 = vreinterpretq_u16_u8(gt_nine);
    let mask8 = vshrn_n_u16::<4>(mask16);
    let mask64 = vget_lane_u64::<0>(vreinterpret_u64_u8(mask8));

    let mut count = need;
    if mask64 != 0 {
        let parsed_digits = (mask64.trailing_zeros() >> 2) as usize;
        if parsed_digits < need {
            count = parsed_digits;
        }
    }

    let sum = match count {
        0 => 0,
        1 => vgetq_lane_u8::<0>(digits) as u64,
        2 => (vgetq_lane_u8::<0>(digits) as u64) * 10 + (vgetq_lane_u8::<1>(digits) as u64),
        3 => {
            let shifted = vextq_u8::<3>(vdupq_n_u8(0), digits);
            let p1 = packadd_1!(shifted);
            (vgetq_lane_u16::<6>(p1) as u64) * 100 + (vgetq_lane_u16::<7>(p1) as u64)
        }
        4 => {
            let shifted = vextq_u8::<4>(vdupq_n_u8(0), digits);
            let p1 = packadd_1!(shifted);
            (vgetq_lane_u16::<6>(p1) as u64) * 100 + (vgetq_lane_u16::<7>(p1) as u64)
        }
        5 => simd_add_5_8!(digits, 5),
        6 => simd_add_5_8!(digits, 6),
        7 => simd_add_5_8!(digits, 7),
        8 => simd_add_5_8!(digits, 8),
        9 => simd_add_9_15!(digits, 9),
        10 => simd_add_9_15!(digits, 10),
        11 => simd_add_9_15!(digits, 11),
        12 => simd_add_9_15!(digits, 12),
        13 => simd_add_9_15!(digits, 13),
        14 => simd_add_9_15!(digits, 14),
        15 => simd_add_9_15!(digits, 15),
        16 => {
            let p = simd_add_8!(digits);
            vgetq_lane_u64::<0>(p) * 100000000 + vgetq_lane_u64::<1>(p)
        }
        _ => core::hint::unreachable_unchecked(),
    };

    (sum, count)
}

#[repr(C, align(16))]
struct ShuffleTable([u8; 16 * 17]);

const NEON_SHUFFLE_TABLE: ShuffleTable = ShuffleTable([
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, // 1
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, // 2
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, // 3
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, // 4
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, // 5
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, // 6
    255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, // 7
    255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, // 8
    255, 255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, // 9
    255, 255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // 10
    255, 255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // 11
    255, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // 12
    255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // 13
    255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, // 14
    255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, // 15
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // 16
]);

#[inline(always)]
pub unsafe fn simd_str2int(c: &[u8], need: usize) -> (u64, usize) {
    simd_str2int_pairwise(c, need)
}

/// vertical addition which might be slower than pairwise addition on neon
#[inline(always)]
pub unsafe fn simd_str2int_vertical(c: &[u8], need: usize) -> (u64, usize) {
    debug_assert!(need <= 16);

    let data = vld1q_u8(c.as_ptr());
    let digits = vsubq_u8(data, vdupq_n_u8(b'0'));

    // shift and shrink to get the mask
    let gt_nine = vcgtq_u8(digits, vdupq_n_u8(9));
    let mask16 = vreinterpretq_u16_u8(gt_nine);
    let mask8 = vshrn_n_u16::<4>(mask16);
    let mask64 = vget_lane_u64::<0>(vreinterpret_u64_u8(mask8));

    let first_non_digit = if mask64 == 0 {
        16
    } else {
        (mask64.trailing_zeros() >> 2) as usize
    };
    let count = if first_non_digit < need {
        first_non_digit
    } else {
        need
    };

    // 2. align to the right using vqtbl1q_u8, which is a 16-lane shuffle. we prepare a shuffle
    //    table to shuffle the digits
    let shuffle_indices = vld1q_u8(NEON_SHUFFLE_TABLE.0.as_ptr().add(count * 16));
    let aligned = vqtbl1q_u8(digits, shuffle_indices);

    // 3. Tree Reduction
    let even8 = vget_low_u8(vuzp1q_u8(aligned, aligned));
    let odd8 = vget_low_u8(vuzp2q_u8(aligned, aligned));
    let res_u16 = vaddw_u8(vmull_u8(even8, vdup_n_u8(10)), odd8);

    let even16 = vget_low_u16(vuzp1q_u16(res_u16, res_u16));
    let odd16 = vget_low_u16(vuzp2q_u16(res_u16, res_u16));
    let res_u32 = vaddw_u16(vmull_n_u16(even16, 100), odd16);

    let even32 = vget_low_u32(vuzp1q_u32(res_u32, res_u32));
    let odd32 = vget_low_u32(vuzp2q_u32(res_u32, res_u32));
    let res_u64 = vaddw_u32(vmull_n_u32(even32, 10000), odd32);

    let high = vgetq_lane_u64::<0>(res_u64);
    let low = vgetq_lane_u64::<1>(res_u64);

    let sum = low + high * 100_000_000;

    (sum, count)
}

/// horizontal addition
#[inline(always)]
pub unsafe fn simd_str2int_pairwise(c: &[u8], need: usize) -> (u64, usize) {
    debug_assert!(need <= 16);

    let data = vld1q_u8(c.as_ptr());
    let digits = vsubq_u8(data, vdupq_n_u8(b'0'));

    let gt_nine = vcgtq_u8(digits, vdupq_n_u8(9));
    let mask16 = vreinterpretq_u16_u8(gt_nine);
    let mask8 = vshrn_n_u16::<4>(mask16);
    let mask64 = vget_lane_u64::<0>(vreinterpret_u64_u8(mask8));

    let first_non_digit = if mask64 == 0 {
        16
    } else {
        (mask64.trailing_zeros() >> 2) as usize
    };
    let count = if first_non_digit < need {
        first_non_digit
    } else {
        need
    };

    // 3. align to the right
    let shuffle_indices = vld1q_u8(NEON_SHUFFLE_TABLE.0.as_ptr().add(count * 16));
    let aligned = vqtbl1q_u8(digits, shuffle_indices);

    // 4. tree reduction using pairwise addition
    // there is no overflow risk because the max sum of 8 digits is 10 * 9 = 90 < 255
    let w1 = vld1q_u8([10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1].as_ptr());
    let m1 = vmulq_u8(aligned, w1);
    let res_u16 = vpaddlq_u8(m1);

    let w2 = vld1q_u16([100, 1, 100, 1, 100, 1, 100, 1].as_ptr());
    let m2 = vmulq_u16(res_u16, w2);
    let res_u32 = vpaddlq_u16(m2);

    let w3 = vld1q_u32([10000, 1, 10000, 1].as_ptr());
    let m3 = vmulq_u32(res_u32, w3);
    let res_u64 = vpaddlq_u32(m3);

    let high = vgetq_lane_u64::<0>(res_u64);
    let low = vgetq_lane_u64::<1>(res_u64);

    let sum = low + high * 100_000_000;

    (sum, count)
}

const NUMBERS: &[&[u8]] = &[
    b"1               ",
    b"12              ",
    b"123             ",
    b"1234            ",
    b"12345           ",
    b"123456          ",
    b"1234567         ",
    b"12345678        ",
    b"123456789       ",
    b"1234567890      ",
    b"12345678901     ",
    b"123456789012    ",
    b"1234567890123   ",
    b"12345678901234  ",
    b"123456789012345 ",
    b"1234567890123456",
];

fn bench_str2int(c: &mut Criterion) {
    let mut group = c.benchmark_group("Str2Int_Fixed_Length");

    for i in 1..=16 {
        let s = NUMBERS[i - 1];
        let len = i;

        group.bench_with_input(BenchmarkId::new("sonic_pairwise", len), &s, |b, &s| {
            b.iter(|| unsafe { black_box(simd_str2int_pairwise(s, len)) })
        });

        group.bench_with_input(BenchmarkId::new("sonic_vertical", len), &s, |b, &s| {
            b.iter(|| unsafe { black_box(simd_str2int_vertical(s, len)) })
        });

        group.bench_with_input(BenchmarkId::new("sonic_vertical_vext", len), &s, |b, &s| {
            b.iter(|| unsafe { black_box(simd_str2int_vertical_vext(s, len)) })
        });

        group.bench_with_input(BenchmarkId::new("atoi_simd", len), &s, |b, &s| {
            b.iter(|| black_box(atoi_simd::parse::<u64, false, false>(s)))
        });
    }
    group.finish();

    let mut rng = rand::rng();
    let random_indices_8: Vec<usize> = (0..1024).map(|_| rng.random_range(0..8)).collect();
    let random_indices_16: Vec<usize> = (0..1024).map(|_| rng.random_range(0..16)).collect();

    let mut group_rnd = c.benchmark_group("Str2Int_Random_Length");

    group_rnd.bench_function("sonic_pairwise_rnd_8", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_8[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_pairwise(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.bench_function("sonic_pairwise_rnd_16", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_16[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_pairwise(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.bench_function("sonic_vertical_rnd_8", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_8[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_vertical(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.bench_function("sonic_vertical_rnd_16", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_16[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_vertical(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.bench_function("sonic_vertical_vext_rnd_8", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_8[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_vertical_vext(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.bench_function("sonic_vertical_vext_rnd_16", |b| {
        let mut i = 0;
        b.iter(|| {
            let idx = random_indices_16[i % 1024];
            i += 1;
            unsafe { black_box(simd_str2int_vertical_vext(NUMBERS[idx], idx + 1)) }
        })
    });

    group_rnd.finish();
}

criterion_group!(benches, bench_str2int);
criterion_main!(benches);
