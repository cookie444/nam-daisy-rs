//! Shared helpers for the desktop parity harness (not shipped to the target).

use serde_json::Value;

/// Parsed `.nam` JSON: ordered weights + a metadata-only JSON blob for `.namb`.
pub struct NamJson {
    pub weights: Vec<f32>,
    pub metadata_json: Vec<u8>,
    pub sample_rate: f32,
    pub input_level_dbu: f32,
    pub output_level_dbu: f32,
    pub head_scale: f32,
}

/// Read a `.nam` JSON file (as produced by the NAM training stack).
pub fn parse_nam(json: &str) -> Result<NamJson, String> {
    let mut root: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let obj = root.as_object_mut().ok_or("nam root is not an object")?;

    if obj.get("architecture").and_then(|v| v.as_str()) != Some("WaveNet") {
        return Err("only WaveNet is supported by the parity harness".into());
    }

    let head_scale = obj
        .get("config")
        .and_then(|c| c.get("head_scale"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let sample_rate = obj
        .get("sample_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(48000.0) as f32;
    let input_level_dbu = obj
        .get("metadata")
        .and_then(|m| m.get("input_level_dbu"))
        .and_then(|v| v.as_f64())
        .unwrap_or(12.0) as f32;
    let output_level_dbu = obj
        .get("metadata")
        .and_then(|m| m.get("output_level_dbu"))
        .and_then(|v| v.as_f64())
        .unwrap_or(-6.0) as f32;

    let weights_val = obj.remove("weights").ok_or("missing weights")?;
    let weights: Vec<f32> = weights_val
        .as_array()
        .ok_or("weights is not an array")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
        .collect();

    // Metadata blob = the whole `.nam` minus the `weights` key.
    let metadata_json = serde_json::to_vec(&root).map_err(|e| e.to_string())?;

    Ok(NamJson {
        weights,
        metadata_json,
        sample_rate,
        input_level_dbu,
        output_level_dbu,
        head_scale,
    })
}

/// Golden fixture: `[u32 N][f32 xN input][f32 xN expected output]`.
pub struct Golden {
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
}

pub fn read_golden(bytes: &[u8]) -> Result<Golden, String> {
    if bytes.len() < 4 {
        return Err("golden too short".into());
    }
    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let need = 4 + n * 8;
    if bytes.len() != need {
        return Err(format!("golden size {} != expected {}", bytes.len(), need));
    }
    let mut input = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);
    for i in 0..n {
        let o = 4 + i * 4;
        input.push(f32::from_le_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]));
        let o = 4 + n * 4 + i * 4;
        expected.push(f32::from_le_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
        ]));
    }
    Ok(Golden { input, expected })
}

/// Error-to-signal ratio: `sum((ref-test)^2) / sum(ref^2)` in f64.
pub fn esr(reference: &[f32], test: &[f32]) -> f64 {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for (r, t) in reference.iter().zip(test.iter()) {
        let r = *r as f64;
        let d = r - *t as f64;
        signal += r * r;
        noise += d * d;
    }
    if signal <= f64::EPSILON {
        if noise <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise / signal
    }
}

pub fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for (r, t) in reference.iter().zip(test.iter()) {
        let r = *r as f64;
        let d = r - *t as f64;
        signal += r * r;
        noise += d * d;
    }
    if noise <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal / noise).log10()
    }
}

pub fn max_abs_err(reference: &[f32], test: &[f32]) -> f32 {
    reference
        .iter()
        .zip(test.iter())
        .map(|(r, t)| (r - t).abs())
        .fold(0.0f32, f32::max)
}

/// Encode a `.nam` into a v2 `Original` `.namb`, returning the byte vector.
pub fn encode_namb(nam: &NamJson) -> Result<Vec<u8>, String> {
    let mut out =
        vec![0u8; nam_namb::HEADER_SIZE + nam.metadata_json.len() + 1 + nam.weights.len() * 4];
    let n = nam_namb::encode::encode(
        &mut out,
        &nam.metadata_json,
        &nam.weights,
        nam.sample_rate,
        nam.input_level_dbu,
        nam.output_level_dbu,
    )
    .map_err(|e| format!("encode: {e:?}"))?;
    out.truncate(n);
    Ok(out)
}
