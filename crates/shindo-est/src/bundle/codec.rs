//! Little-endian reading and writing of the values in a model file.

use crate::bundle::FormatError;

/// Converts a length read from a file to `usize`, which has 32 bits on
/// wasm32.
pub(crate) fn to_usize(value: u64) -> Result<usize, FormatError> {
    usize::try_from(value).map_err(|_| FormatError::Truncated)
}

/// Reads values from the front of a byte slice.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    #[must_use]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Returns the number of bytes not read yet.
    #[must_use]
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len()
    }

    /// Checks that every byte has been read.
    pub(crate) fn finish(self) -> Result<(), FormatError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(FormatError::TrailingBytes)
        }
    }

    pub(crate) fn bytes(&mut self, len: usize) -> Result<&'a [u8], FormatError> {
        let (head, rest) = self
            .bytes
            .split_at_checked(len)
            .ok_or(FormatError::Truncated)?;
        self.bytes = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], FormatError> {
        let (head, rest) = self
            .bytes
            .split_first_chunk()
            .ok_or(FormatError::Truncated)?;
        self.bytes = rest;
        Ok(*head)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(u8::from_le_bytes(self.array()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, FormatError> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, FormatError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    pub(crate) fn u8s(&mut self, count: usize) -> Result<Vec<u8>, FormatError> {
        Ok(self.bytes(count)?.to_vec())
    }

    pub(crate) fn u32s(&mut self, count: usize) -> Result<Vec<u32>, FormatError> {
        self.values(count, Self::u32)
    }

    pub(crate) fn i32s(&mut self, count: usize) -> Result<Vec<i32>, FormatError> {
        self.values(count, Self::i32)
    }

    pub(crate) fn f64s(&mut self, count: usize) -> Result<Vec<f64>, FormatError> {
        self.values(count, Self::f64)
    }

    /// Reads `count` values with `read`, checking that the bytes hold them all
    /// before allocating, so that a corrupt count cannot ask for a huge
    /// allocation.
    fn values<T>(
        &mut self,
        count: usize,
        read: fn(&mut Self) -> Result<T, FormatError>,
    ) -> Result<Vec<T>, FormatError> {
        let fits = count
            .checked_mul(size_of::<T>())
            .is_some_and(|len| len <= self.bytes.len());
        if !fits {
            return Err(FormatError::Truncated);
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(read(self)?);
        }
        Ok(values)
    }
}

/// Writes values one after another.
#[derive(Default)]
pub(crate) struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the bytes written.
    #[must_use]
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.bytes(&value.to_le_bytes());
    }

    pub(crate) fn u8s(&mut self, values: &[u8]) {
        self.bytes(values);
    }

    pub(crate) fn u32s(&mut self, values: &[u32]) {
        for &value in values {
            self.u32(value);
        }
    }

    pub(crate) fn i32s(&mut self, values: &[i32]) {
        for &value in values {
            self.i32(value);
        }
    }

    pub(crate) fn f64s(&mut self, values: &[f64]) {
        for &value in values {
            self.f64(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_the_writer_wrote() {
        // A NaN with a payload, which a canonicalizing writer would lose.
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let mut writer = Writer::new();
        writer.u8(0xab);
        writer.u32(0x0102_0304);
        writer.i32(-5);
        writer.u64(u64::MAX - 1);
        writer.f64(nan);
        writer.f64(-0.0);
        writer.u8s(&[1, 2]);
        writer.u32s(&[3, u32::MAX]);
        writer.i32s(&[-1, i32::MIN]);
        writer.f64s(&[0.5, f64::MAX, nan]);
        let bytes = writer.into_bytes();

        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.u8(), Ok(0xab));
        assert_eq!(reader.u32(), Ok(0x0102_0304));
        assert_eq!(reader.i32(), Ok(-5));
        assert_eq!(reader.u64(), Ok(u64::MAX - 1));
        let read_nan = reader.f64().expect("the NaN should be read");
        assert_eq!(read_nan.to_bits(), nan.to_bits());
        let zero = reader.f64().expect("the zero should be read");
        assert_eq!(zero.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(reader.u8s(2), Ok(vec![1, 2]));
        assert_eq!(reader.u32s(2), Ok(vec![3, u32::MAX]));
        assert_eq!(reader.i32s(2), Ok(vec![-1, i32::MIN]));
        let floats = reader.f64s(3).expect("the floats should be read");
        let bits: Vec<u64> = floats.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, [0.5_f64.to_bits(), f64::MAX.to_bits(), nan.to_bits()]);
        assert_eq!(reader.remaining(), 0);
        assert_eq!(reader.finish(), Ok(()));
    }

    #[test]
    fn writes_little_endian() {
        let mut writer = Writer::new();
        writer.u32(0x0102_0304);
        assert_eq!(writer.into_bytes(), [4, 3, 2, 1]);
    }

    #[test]
    fn refuses_to_read_past_the_end() {
        let mut reader = Reader::new(&[1, 2, 3]);
        assert_eq!(reader.u32(), Err(FormatError::Truncated));
    }

    #[test]
    fn rejects_an_array_longer_than_the_bytes() {
        let bytes = [0; 15];
        assert_eq!(
            Reader::new(&bytes).f64s(usize::MAX),
            Err(FormatError::Truncated)
        );
        assert_eq!(Reader::new(&bytes).f64s(2), Err(FormatError::Truncated));
        assert_eq!(Reader::new(&bytes).bytes(16), Err(FormatError::Truncated));
    }

    #[test]
    fn finish_rejects_leftover_bytes() {
        let mut reader = Reader::new(&[1, 2]);
        assert_eq!(reader.u8(), Ok(1));
        assert_eq!(reader.remaining(), 1);
        assert_eq!(reader.finish(), Err(FormatError::TrailingBytes));
    }
}
