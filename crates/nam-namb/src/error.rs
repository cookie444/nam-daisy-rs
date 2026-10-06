//! Typed `.namb` errors. Mnemonics mirror `docs/namb-spec.md` §8.

/// Errors produced while parsing or validating a `.namb` buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NambError {
    /// File shorter than 80 bytes, or trailing bytes do not complete an f32.
    Truncated { got: usize, need: usize },
    /// Magic is not `0x4E414D42`.
    InvalidMagic(u32),
    /// Version is not 1 or 2.
    InvalidVersion(u16),
    /// `weights_offset` exceeds the file size.
    WeightsOffsetOutOfBounds { offset: u32, file_len: usize },
    /// `weights_offset` < 80.
    InvalidWeightsOffset { offset: u32, header_size: usize },
    /// Calculated CRC != stored CRC.
    CrcMismatch { got: u32, expected: u32 },
    /// v2+ file without `FLAG_HAS_CRC32`.
    CrcMissing { version: u16 },
    /// Legacy v1 file with the `crc32 == 0` sentinel rejected by policy.
    CrcMissingV1,
    /// Non-finite weight in the binary weight section.
    NonFiniteWeight { index: usize },
    /// Non-finite header metadata float.
    InvalidHeaderField,
    /// Metadata section is not valid UTF-8 / not parseable as a NAM model.
    MetadataJson,
    /// Topology referenced by the metadata is unsupported by this crate.
    UnsupportedTopology,
    /// The weight block does not match the topology's expected element count.
    WeightCountMismatch { got: usize, need: usize },
    /// Output buffer supplied to the encoder is too small.
    EncodeBufferTooSmall { got: usize, need: usize },
}
