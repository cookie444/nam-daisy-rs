//! `.namb` 80-byte packed header, validated without alignment assumptions.

use crate::crc::{crc32_ieee, Crc32};
use crate::error::NambError;

/// Magic `0x4E414D42` ("NAMB" little-endian).
pub const MAGIC: u32 = 0x4E41_4D42;
/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 80;
/// `flags` bit 0: header `crc32` is present and must be verified.
pub const FLAG_HAS_CRC32: u8 = 0x01;

/// Weight layout selector (active for `version >= 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Layout {
    /// Weights exactly as they appear in the `.nam` JSON.
    Original = 0,
    /// LSTM gate-major transposed layout.
    GateMajorLstm = 1,
    /// WaveNet interleaved-4 conv layout with transposed dense layers.
    Interleaved4WaveNet = 2,
}

impl Layout {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Layout::GateMajorLstm,
            2 => Layout::Interleaved4WaveNet,
            _ => Layout::Original,
        }
    }
}

#[inline]
fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
#[inline]
fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[inline]
fn read_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Validated view of the fixed header fields.
#[derive(Debug, Clone, Copy)]
pub struct NambHeader {
    pub magic: u32,
    pub version: u16,
    pub layout: Layout,
    pub flags: u8,
    pub weights_offset: u32,
    pub crc32: u32,
    pub sample_rate: f32,
    pub input_level_dbu: f32,
    pub output_level_dbu: f32,
}

impl NambHeader {
    /// Parse and structurally validate the header fields.
    ///
    /// Does **not** verify the CRC; call [`NambHeader::validate_integrity`].
    pub fn parse(bytes: &[u8]) -> Result<Self, NambError> {
        if bytes.len() < HEADER_SIZE {
            return Err(NambError::Truncated {
                got: bytes.len(),
                need: HEADER_SIZE,
            });
        }
        let magic = read_u32(bytes, 0);
        if magic != MAGIC {
            return Err(NambError::InvalidMagic(magic));
        }
        let version = read_u16(bytes, 4);
        if version != 1 && version != 2 {
            return Err(NambError::InvalidVersion(version));
        }
        // v1: byte 7 is part of the reserved region; layout is Original.
        let (layout, flags) = if version >= 2 {
            (Layout::from_u8(bytes[6]), bytes[7])
        } else {
            (Layout::Original, 0)
        };

        let weights_offset = read_u32(bytes, 12);
        if (weights_offset as usize) < HEADER_SIZE {
            return Err(NambError::InvalidWeightsOffset {
                offset: weights_offset,
                header_size: HEADER_SIZE,
            });
        }
        if weights_offset as usize > bytes.len() {
            return Err(NambError::WeightsOffsetOutOfBounds {
                offset: weights_offset,
                file_len: bytes.len(),
            });
        }

        let sample_rate = read_f32(bytes, 64);
        let input_level_dbu = read_f32(bytes, 68);
        let output_level_dbu = read_f32(bytes, 72);
        for f in [sample_rate, input_level_dbu, output_level_dbu] {
            if !f.is_finite() {
                return Err(NambError::InvalidHeaderField);
            }
        }

        Ok(Self {
            magic,
            version,
            layout,
            flags,
            weights_offset,
            crc32: read_u32(bytes, 24),
            sample_rate,
            input_level_dbu,
            output_level_dbu,
        })
    }

    /// Verify CRC32 according to the version-dependent coverage rules.
    pub fn validate_integrity(&self, bytes: &[u8]) -> Result<(), NambError> {
        let stored = self.crc32;
        if self.version >= 2 {
            if self.flags & FLAG_HAS_CRC32 == 0 {
                return Err(NambError::CrcMissing {
                    version: self.version,
                });
            }
            // Covers everything except the 4-byte crc32 field at offset 24.
            let mut c = Crc32::new();
            c.update(&bytes[..24]);
            c.update(&bytes[28..]);
            let got = c.finalize();
            if got != stored {
                return Err(NambError::CrcMismatch {
                    got,
                    expected: stored,
                });
            }
        } else {
            if stored == 0 {
                return Err(NambError::CrcMissingV1);
            }
            let off = self.weights_offset as usize;
            let got = crc32_ieee(&bytes[off..]);
            if got != stored {
                return Err(NambError::CrcMismatch {
                    got,
                    expected: stored,
                });
            }
        }
        Ok(())
    }
}
