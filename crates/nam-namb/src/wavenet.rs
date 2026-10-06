//! Extract a WaveNet A1 topology from NAM metadata JSON.
//!
//! Supports the two-layer WaveNet "A1" catalog shape (Nano/Feather/Lite/
//! Standard). Gated/FiLM/A2 layers are rejected as unsupported.

use crate::json::{Json, JsonError};

pub const MAX_LAYERS: usize = 2;
pub const MAX_DILATIONS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Tanh,
    Relu,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerTopo {
    pub input_size: u16,
    pub condition_size: u16,
    pub head_size: u16,
    pub channels: u16,
    pub kernel_size: u16,
    pub dilations: [u16; MAX_DILATIONS],
    pub num_dilations: usize,
    pub gated: bool,
    pub head_bias: bool,
    pub activation: Activation,
}

impl Default for LayerTopo {
    fn default() -> Self {
        Self {
            input_size: 1,
            condition_size: 1,
            head_size: 1,
            channels: 1,
            kernel_size: 1,
            dilations: [0; MAX_DILATIONS],
            num_dilations: 0,
            gated: false,
            head_bias: false,
            activation: Activation::Tanh,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WavenetTopo {
    pub layers: [LayerTopo; MAX_LAYERS],
    pub num_layers: usize,
    pub head_scale: f32,
}

impl Default for WavenetTopo {
    fn default() -> Self {
        Self {
            layers: [LayerTopo::default(); MAX_LAYERS],
            num_layers: 0,
            head_scale: 1.0,
        }
    }
}

/// Parse NAM metadata JSON into a WaveNet topology.
pub fn parse_wavenet_json(meta: &[u8]) -> Result<WavenetTopo, JsonError> {
    let mut j = Json::new(meta);
    let mut topo = WavenetTopo::default();

    if j.peek_non_ws()? != b'{' {
        return Err(JsonError::Expected("object"));
    }
    j.expect(b'{')?;
    if j.peek_non_ws()? == b'}' {
        return Err(JsonError::Expected("architecture"));
    }
    loop {
        let key = j.parse_string()?;
        j.expect(b':')?;
        match key {
            b"architecture" => {
                let arch = j.parse_string()?;
                if arch != b"WaveNet" {
                    return Err(JsonError::Expected("WaveNet"));
                }
            }
            b"config" => {
                parse_config(&mut j, &mut topo)?;
            }
            b"head_scale" => {
                topo.head_scale = j.parse_f32()?;
            }
            _ => j.skip_value()?,
        }
        match j.peek_non_ws()? {
            b',' => {
                j.bump_comma();
            }
            b'}' => {
                j.bump_comma();
                break;
            }
            _ => return Err(JsonError::Expected("',' or '}'")),
        }
    }

    if topo.num_layers == 0 {
        return Err(JsonError::Expected("layers"));
    }
    Ok(topo)
}

fn parse_config(j: &mut Json, topo: &mut WavenetTopo) -> Result<(), JsonError> {
    j.expect(b'{')?;
    if j.peek_non_ws()? == b'}' {
        j.bump_comma();
        return Ok(());
    }
    loop {
        let key = j.parse_string()?;
        j.expect(b':')?;
        match key {
            b"layers" => parse_layers(j, topo)?,
            b"head_scale" => topo.head_scale = j.parse_f32()?,
            _ => j.skip_value()?,
        }
        match j.peek_non_ws()? {
            b',' => j.bump_comma(),
            b'}' => {
                j.bump_comma();
                break;
            }
            _ => return Err(JsonError::Expected("',' or '}'")),
        }
    }
    Ok(())
}

fn parse_layers(j: &mut Json, topo: &mut WavenetTopo) -> Result<(), JsonError> {
    j.expect(b'[')?;
    if j.peek_non_ws()? == b']' {
        j.bump_comma();
        return Ok(());
    }
    loop {
        if topo.num_layers >= MAX_LAYERS {
            return Err(JsonError::Expected("<= 2 layers"));
        }
        let idx = topo.num_layers;
        let mut layer = LayerTopo::default();
        parse_layer(j, &mut layer)?;
        topo.layers[idx] = layer;
        topo.num_layers += 1;

        match j.peek_non_ws()? {
            b',' => j.bump_comma(),
            b']' => {
                j.bump_comma();
                break;
            }
            _ => return Err(JsonError::Expected("',' or ']'")),
        }
    }
    Ok(())
}

fn parse_layer(j: &mut Json, layer: &mut LayerTopo) -> Result<(), JsonError> {
    j.expect(b'{')?;
    if j.peek_non_ws()? == b'}' {
        j.bump_comma();
        return Ok(());
    }
    loop {
        let key = j.parse_string()?;
        j.expect(b':')?;
        match key {
            b"input_size" => layer.input_size = j.parse_u16()?,
            b"condition_size" => layer.condition_size = j.parse_u16()?,
            b"head_size" => layer.head_size = j.parse_u16()?,
            b"channels" => layer.channels = j.parse_u16()?,
            b"kernel_size" => layer.kernel_size = j.parse_u16()?,
            b"dilations" => parse_dilations(j, layer)?,
            b"activation" => {
                let a = j.parse_string()?;
                layer.activation = match a {
                    b"Tanh" => Activation::Tanh,
                    b"ReLU" => Activation::Relu,
                    _ => return Err(JsonError::Expected("Tanh or ReLU")),
                };
            }
            b"gated" => layer.gated = j.parse_bool()?,
            b"head_bias" => layer.head_bias = j.parse_bool()?,
            _ => j.skip_value()?,
        }
        match j.peek_non_ws()? {
            b',' => j.bump_comma(),
            b'}' => {
                j.bump_comma();
                break;
            }
            _ => return Err(JsonError::Expected("',' or '}'")),
        }
    }
    Ok(())
}

fn parse_dilations(j: &mut Json, layer: &mut LayerTopo) -> Result<(), JsonError> {
    j.expect(b'[')?;
    if j.peek_non_ws()? == b']' {
        j.bump_comma();
        return Ok(());
    }
    loop {
        if layer.num_dilations >= MAX_DILATIONS {
            return Err(JsonError::Expected("fewer dilations"));
        }
        layer.dilations[layer.num_dilations] = j.parse_u16()?;
        layer.num_dilations += 1;
        match j.peek_non_ws()? {
            b',' => j.bump_comma(),
            b']' => {
                j.bump_comma();
                break;
            }
            _ => return Err(JsonError::Expected("',' or ']'")),
        }
    }
    Ok(())
}
