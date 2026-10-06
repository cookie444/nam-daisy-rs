//! CRC-32/ISO-HDLC (IEEE 802.3): poly 0xEDB88320, init/xorout 0xFFFFFFFF.

const POLY: u32 = 0xEDB8_8320;

/// Incremental CRC state.
#[derive(Clone, Copy)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    #[inline]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &b in data {
            crc ^= b as u32;
            let mut i = 0;
            while i < 8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (POLY & mask);
                i += 1;
            }
        }
        self.state = crc;
    }

    #[inline]
    pub const fn finalize(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot CRC-32 over a contiguous slice.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finalize()
}
