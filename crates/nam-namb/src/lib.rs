//! `no_std` parser for the `.namb` Neural Amp Modeler binary format.
//!
//! Pure-Rust, allocation-free. Byte layout follows `docs/namb-spec.md`
//! (see repo `docs/CONTRACTS.md`). All multi-byte fields are little-endian.
//!
//! The parser is split into two concerns:
//!  * [`Namb::parse`] — header validation + CRC32 integrity + weight slicing.
//!  * [`wavenet::parse_wavenet_json`] — topology extraction from the optional
//!    metadata JSON, sufficient to build a WaveNet A1 engine without `alloc`.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod crc;
pub mod encode;
pub mod error;
pub mod header;
pub mod json;
pub mod wavenet;

pub use error::NambError;
pub use header::{Layout, NambHeader, FLAG_HAS_CRC32, HEADER_SIZE, MAGIC};
pub use wavenet::{LayerTopo, WavenetTopo, MAX_DILATIONS, MAX_LAYERS};

/// A borrowed, validated view over a `.namb` byte buffer.
#[derive(Clone, Copy)]
pub struct Namb<'a> {
    header: NambHeader,
    metadata: &'a [u8],
    weights: &'a [u8],
}

impl<'a> Namb<'a> {
    /// Validate magic, version, offsets and CRC32. Metadata is *not* parsed here.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, NambError> {
        let header = NambHeader::parse(bytes)?;
        header.validate_integrity(bytes)?;

        let off = header.weights_offset as usize;
        // `header.validate_integrity` already guarantees HEADER_SIZE <= off <= len.
        let (meta, weights) = bytes.split_at(off);
        // Metadata occupies bytes [HEADER_SIZE..off).
        let metadata = meta.get(HEADER_SIZE..).unwrap_or(&[]);

        // Weights must be a whole number of f32.
        if weights.len() % 4 != 0 {
            return Err(NambError::Truncated {
                got: weights.len(),
                need: weights.len() - (weights.len() % 4) + 4,
            });
        }
        Ok(Self {
            header,
            metadata,
            weights,
        })
    }

    pub fn header(&self) -> &NambHeader {
        &self.header
    }

    /// Raw metadata bytes between the header and the weights block.
    pub fn metadata_bytes(&self) -> &'a [u8] {
        self.metadata
    }

    /// Raw little-endian weights block.
    pub fn weights_bytes(&self) -> &'a [u8] {
        self.weights
    }

    pub fn num_weights(&self) -> usize {
        self.weights.len() / 4
    }

    /// Iterate the weights as `f32` without allocation.
    pub fn weights(&self) -> WeightIter<'a> {
        WeightIter {
            bytes: self.weights,
        }
    }

    /// Parse the metadata JSON into a WaveNet A1 topology.
    pub fn wavenet_topology(&self) -> Result<WavenetTopo, NambError> {
        let meta = json::metadata_slice(self.metadata);
        wavenet::parse_wavenet_json(meta).map_err(|_| NambError::MetadataJson)
    }
}

/// Zero-allocation iterator over little-endian `f32` weights.
pub struct WeightIter<'a> {
    bytes: &'a [u8],
}

impl Iterator for WeightIter<'_> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        if self.bytes.len() < 4 {
            return None;
        }
        let (head, rest) = self.bytes.split_at(4);
        self.bytes = rest;
        let b = [head[0], head[1], head[2], head[3]];
        Some(f32::from_le_bytes(b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bytes.len() / 4;
        (n, Some(n))
    }
}
