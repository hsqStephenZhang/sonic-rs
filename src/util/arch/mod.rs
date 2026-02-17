use crate::reader::{Reader, ReaderExt};
use crate::parser::is_whitespace;

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq", target_feature = "avx2", target_feature = "sse2"))] {
        mod x86_64;
        pub use x86_64::*;
    } else if #[cfg(all(target_feature="sve2", target_arch="aarch64"))] {
        mod sve2;
        pub use sve2::*;
    } else if #[cfg(all(target_feature="neon", target_arch="aarch64"))] {
        mod aarch64;
        pub use aarch64::*;
    } else {
        mod fallback;
        pub use fallback::*;
    }
}

#[inline(always)]
fn skip_exponent<'de, R: Reader<'de>>(reader: &mut R) -> crate::Result<()> {
    reader.eat_if(|ch| ch == b'e' || ch == b'E');
    skip_single_digit(reader)?;
    while reader.eat_if(|ch| matches!(ch, b'0'..=b'9')) {}
    Ok(())
}

#[inline(always)]
fn skip_single_digit<'de, R: Reader<'de>>(reader: &mut R) -> crate::Result<u8> {
    if let Some(ch) = reader.next() {
        if !ch.is_ascii_digit() {
            todo!()
        } else {
            Ok(ch)
        }
    } else {
        todo!()
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_feature = "sve2"))] {
        pub(crate) struct SpaceSkipper;

        impl SpaceSkipper {
            pub fn new() -> Self {
                Self
            }

            #[inline(always)]
            pub(crate) fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
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
                        return Some(ch);
                    }
                }
                None
            }
        }
    } else {
        use sonic_simd::{i8x32, m8x32, u8x32, u8x64, Mask, Simd};

        pub(crate) struct SpaceSkipper {
            nospace_bits: u64,    // SIMD marked nospace bitmap
            nospace_start: isize, // the start position of nospace_bits
        }

        impl SpaceSkipper {
            pub(crate) fn new() -> Self {
                Self {
                    nospace_bits: 0,
                    nospace_start: -128,
                }
            }

            #[inline(always)]
            pub(crate) fn skip_space<'de, R: Reader<'de>>(&mut self, reader: &mut R) -> Option<u8> {
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
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_non_space_bits() {
        let input = b"\t\r\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        cfg_if::cfg_if! {
            if #[cfg(all(target_feature="sve2", target_arch="aarch64"))] {
                let first_nonspace_idx = unsafe { get_nonspace_index(std::mem::transmute(input)) };
                // sve2 cannot generate the full bitmap(without performance loss)
                assert_eq!(first_nonspace_idx, 4, "first non-space index is {first_nonspace_idx}");
            } else {
                let non_space_bits = unsafe { get_nonspace_bits(input) };
                let expected_bits = 0b1111111111111111111111111111111111111111111111111111111111110000;
                assert_eq!(non_space_bits, expected_bits, "bits is {non_space_bits:b}");
            }
        }
    }
}
