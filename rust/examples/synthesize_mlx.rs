//! Generate Tamil speech from the MLX-trained ONNX model.
//!
//!   cargo run --release --example synthesize_mlx -- "வணக்கம்" out.wav ../models/tamil_mlx.onnx

use anyhow::Result;
use tamil_tts::mlx_tts::MlxTts;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).map(|s| s.as_str()).unwrap_or("வணக்கம்");
    let out = args.get(2).map(|s| s.as_str()).unwrap_or("out_mlx.wav");
    let model = args.get(3).map(|s| s.as_str()).unwrap_or("../models/tamil_mlx.onnx");

    let mut tts = MlxTts::from_file(model)?;
    tts.save(text, out, 800)?;
    println!("wrote {out} @ {} Hz", tts.sample_rate());
    Ok(())
}
