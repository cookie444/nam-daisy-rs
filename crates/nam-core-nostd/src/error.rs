//! Engine construction errors.

use nam_namb::NambError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineError {
    /// Underlying `.namb` container is invalid.
    Namb(NambError),
    /// Topology is not a supported WaveNet A1 shape.
    UnsupportedTopology,
    /// Weight count does not match the topology.
    WeightCountMismatch { got: usize, need: usize },
    /// Geometry exceeds the compiled fixed capacities.
    CapacityExceeded,
    /// Parameter block supplied to `process` is malformed.
    BadParameter,
}

impl From<NambError> for EngineError {
    fn from(e: NambError) -> Self {
        EngineError::Namb(e)
    }
}
