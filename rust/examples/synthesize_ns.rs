//! Generate Tamil speech from the non-AR ONNX model (two graphs, single forward).
//!
//!   cargo run --release --example synthesize_ns -- "வணக்கம்" out.wav ../models/tamil_ns

use anyhow::Result;
use tamil_tts::mlx_ns_tts::MlxNsTts;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).map(|s| s.as_str()).unwrap_or("வணக்கம்");
    let out = args.get(2).map(|s| s.as_str()).unwrap_or("out_ns.wav");
    let prefix = args.get(3).map(|s| s.as_str()).unwrap_or("../models/tamil_ns");
    let mut tts = MlxNsTts::from_prefix(prefix)?;
    tts.save(text, out, 1.0)?;
    println!("wrote {out} @ {} Hz", tts.sample_rate());
    Ok(())
}
