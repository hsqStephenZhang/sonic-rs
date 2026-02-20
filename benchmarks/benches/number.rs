use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use sonic_number::{simd_str2int_pairwise, simd_str2int_vertical, simd_str2int_vertical_vext};

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
