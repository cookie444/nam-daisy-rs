//! Convert a `.nam` JSON model into a `.namb` (v2, Original layout) container.
//!
//! This is a development convenience for the parity harness and for staging
//! models on the Daisy Seed. The production converter is Tone3000's.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: nam-to-namb <input.nam> <output.namb>");
        std::process::exit(2);
    }
    if let Err(e) = run(&args[1], &args[2]) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(input: &str, output: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(input).map_err(|e| e.to_string())?;
    let mut root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;

    let obj = root.as_object_mut().ok_or("root is not an object")?;
    let arch = obj
        .get("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if arch != "WaveNet" {
        return Err(format!("unsupported architecture: {arch}"));
    }
    let sample_rate = obj
        .get("sample_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(48000.0) as f32;
    let input_dbu = obj
        .get("metadata")
        .and_then(|m| m.get("input_level_dbu"))
        .and_then(|v| v.as_f64())
        .unwrap_or(12.0) as f32;
    let output_dbu = obj
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

    let metadata = serde_json::to_vec(&root).map_err(|e| e.to_string())?;

    let mut buf = vec![0u8; nam_namb::HEADER_SIZE + metadata.len() + 1 + weights.len() * 4];
    let n = nam_namb::encode::encode(
        &mut buf,
        &metadata,
        &weights,
        sample_rate,
        input_dbu,
        output_dbu,
    )
    .map_err(|e| format!("encode: {e:?}"))?;

    let mut f = std::fs::File::create(output).map_err(|e| e.to_string())?;
    f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    eprintln!("wrote {output}: {n} bytes, {} weights", weights.len());
    Ok(())
}
