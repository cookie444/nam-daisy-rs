use nam_namb::{crc::crc32_ieee, encode::encode, Layout, Namb, NambError, HEADER_SIZE, MAGIC};

const META: &[u8] = br#"{"architecture":"WaveNet","config":{"layers":[{"input_size":1,"condition_size":1,"head_size":1,"channels":4,"kernel_size":3,"dilations":[1,2],"activation":"ReLU","gated":false,"head_bias":false}],"head_scale":0.02}}"#;

fn build(weights: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_SIZE + META.len() + 1 + weights.len() * 4];
    let n = encode(&mut buf, META, weights, 48000.0, 12.0, -6.0).unwrap();
    buf.truncate(n);
    buf
}

#[test]
fn header_magic_and_version() {
    let w: Vec<f32> = (1..20).map(|i| i as f32).collect();
    let buf = build(&w);
    let h = nam_namb::NambHeader::parse(&buf).unwrap();
    assert_eq!(h.magic, MAGIC);
    assert_eq!(h.version, 2);
    assert_eq!(h.layout, Layout::Original);
    assert_eq!(h.weights_offset as usize, HEADER_SIZE + META.len() + 1);
    assert_eq!(h.sample_rate, 48000.0);
}

#[test]
fn crc_and_weights_roundtrip() {
    let w: Vec<f32> = (0..50).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let buf = build(&w);
    let n = Namb::parse(&buf).unwrap();
    assert_eq!(n.num_weights(), w.len());
    let got: Vec<f32> = n.weights().collect();
    assert_eq!(got, w);
    assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
}

#[test]
fn corrupt_weight_fails_crc() {
    let w: Vec<f32> = vec![1.0; 10];
    let mut buf = build(&w);
    let last = buf.len() - 1;
    buf[last] ^= 0xFF;
    assert!(matches!(
        Namb::parse(&buf),
        Err(NambError::CrcMismatch { .. })
    ));
}

#[test]
fn bad_magic_rejected() {
    let w: Vec<f32> = vec![1.0; 4];
    let mut buf = build(&w);
    buf[0] = 0;
    assert!(matches!(Namb::parse(&buf), Err(NambError::InvalidMagic(_))));
}

#[test]
fn short_buffer_rejected() {
    let buf = [0u8; 10];
    assert!(matches!(
        Namb::parse(&buf),
        Err(NambError::Truncated { .. })
    ));
}

#[test]
fn topology_parsed_from_metadata() {
    let w: Vec<f32> = vec![0.0; 8];
    let buf = build(&w);
    let n = Namb::parse(&buf).unwrap();
    let topo = n.wavenet_topology().unwrap();
    assert_eq!(topo.num_layers, 1);
    assert_eq!(topo.layers[0].channels, 4);
    assert_eq!(topo.layers[0].kernel_size, 3);
    assert_eq!(topo.layers[0].num_dilations, 2);
    assert_eq!(topo.head_scale, 0.02);
    assert_eq!(
        topo.layers[0].activation,
        nam_namb::wavenet::Activation::Relu
    );
}

#[test]
fn v1_layout_is_original_and_crc_over_weights() {
    // Manually downgrade the encoded v2 header to v1: layout ignored, CRC over
    // weights only. This mirrors legacy containers.
    let w: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let buf = build(&w);
    let off = 80 + META.len() + 1;
    let mut v1 = buf.clone();
    v1[4..6].copy_from_slice(&1u16.to_le_bytes());
    v1[7] = 0; // no flags; byte 6 is reserved in v1
    let crc = crc32_ieee(&v1[off..]);
    v1[24..28].copy_from_slice(&crc.to_le_bytes());
    let n = Namb::parse(&v1).unwrap();
    assert_eq!(n.header().layout, Layout::Original);
    assert_eq!(n.num_weights(), w.len());
}
