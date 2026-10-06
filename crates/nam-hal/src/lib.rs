//! `no_std` model storage abstraction for NAM engines.
//!
//! The engine only ever needs a contiguous byte view of a `.namb` blob. On the
//! Daisy Seed that view may live in:
//!  * internal AXI SRAM (copied at boot),
//!  * memory-mapped QSPI flash (`0x9000_0000`), or
//!  * an SD card read into a fixed staging buffer.
//!
//! None of these require heap allocation. [`ModelSource`] is a *positional*
//! reader (so a source can stream into a staging buffer) plus an optional
//! memory-mapped fast path.

#![no_std]

/// Errors a [`ModelSource`] may report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceError {
    /// The model is larger than the destination/staging buffer.
    TooLarge { size: usize, capacity: usize },
    /// The underlying medium failed or is absent.
    Io,
    /// The model is not present at the requested slot.
    NotFound,
}

/// A source of `.namb` model bytes.
pub trait ModelSource {
    /// Byte length of the currently-selected model.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read `dst.len()` bytes starting at `offset`.
    ///
    /// Implementations must return [`SourceError::Io`] rather than panic when
    /// `offset + dst.len() > self.len()`.
    fn read(&mut self, offset: usize, dst: &mut [u8]) -> Result<(), SourceError>;

    /// Optional zero-copy fast path. Returning `Some` lets the caller build the
    /// engine directly over memory-mapped flash without staging.
    ///
    /// The default implementation returns `None`.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }

    /// Copy the whole model into `staging`, returning the used prefix.
    fn load_into<'a>(&mut self, staging: &'a mut [u8]) -> Result<&'a [u8], SourceError> {
        let n = self.len();
        if n > staging.len() {
            return Err(SourceError::TooLarge {
                size: n,
                capacity: staging.len(),
            });
        }
        self.read(0, &mut staging[..n])?;
        Ok(&staging[..n])
    }
}

/// A source backed by an in-memory byte slice (RAM / memory-mapped QSPI).
pub struct SliceSource<'a> {
    bytes: &'a [u8],
}

impl<'a> SliceSource<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
}

impl ModelSource for SliceSource<'_> {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn read(&mut self, offset: usize, dst: &mut [u8]) -> Result<(), SourceError> {
        let end = offset.checked_add(dst.len()).ok_or(SourceError::Io)?;
        let src = self.bytes.get(offset..end).ok_or(SourceError::Io)?;
        dst.copy_from_slice(src);
        Ok(())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(self.bytes)
    }
}
