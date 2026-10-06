//! Minimal `.namb` encoder (used by tests and desktop tooling).
//!
//! Writes a v2, `Original`-layout container: header + null-terminated metadata
//! JSON + contiguous little-endian f32 weights, with a valid CRC32.

use crate::crc::{crc32_ieee, Crc32};
use crate::error::NambError;
use crate::header::{FLAG_HAS_CRC32, HEADER_SIZE, MAGIC};

fn put_u16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut [u8], off: usize, v: f32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Encode a v2 `Original` `.namb` into `out`, returning the total byte length.
///
/// `metadata_json` is written verbatim followed by a NUL terminator.
pub fn encode(
    out: &mut [u8],
    metadata_json: &[u8],
    weights: &[f32],
    sample_rate: f32,
    input_level_dbu: f32,
    output_level_dbu: f32,
) -> Result<usize, NambError> {
    let weights_offset = HEADER_SIZE + metadata_json.len() + 1;
    let total = weights_offset + weights.len() * 4;
    if out.len() < total {
        return Err(NambError::EncodeBufferTooSmall {
            got: out.len(),
            need: total,
        });
    }

    // Header
    for b in out[..HEADER_SIZE].iter_mut() {
        *b = 0;
    }
    put_u32(out, 0, MAGIC);
    put_u16(out, 4, 2);
    out[6] = 0; // Original
    out[7] = FLAG_HAS_CRC32;
    put_u32(out, 12, weights_offset as u32);
    // crc32 field (offset 24) filled after computing.
    let vs = b"NAMB v2 (Original)\0";
    out[32..32 + vs.len()].copy_from_slice(vs);
    put_f32(out, 64, sample_rate);
    put_f32(out, 68, input_level_dbu);
    put_f32(out, 72, output_level_dbu);

    // Metadata + NUL
    out[HEADER_SIZE..HEADER_SIZE + metadata_json.len()].copy_from_slice(metadata_json);
    out[HEADER_SIZE + metadata_json.len()] = 0;

    // Weights
    for (i, w) in weights.iter().enumerate() {
        put_f32(out, weights_offset + i * 4, *w);
    }

    // CRC32 over [..24] + [28..total]
    let mut c = Crc32::new();
    c.update(&out[..24]);
    c.update(&out[28..total]);
    put_u32(out, 24, c.finalize());

    Ok(total)
}

/// Convenience: full-buffer CRC used by tests.
pub fn crc_of(buf: &[u8]) -> u32 {
    crc32_ieee(buf)
}
