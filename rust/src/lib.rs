//! tamil-tts Rust SDK: synthesize Tamil speech on CPU from the non-AR ONNX model.
//!
//! See [`mlx_ns_tts::MlxNsTts`] for the synthesizer (enc_dur + decoder + HiFi-GAN graphs)
//! and [`normalize`] for the text front-end (acronyms/symbols/digits -> Tamil words).
//!
//! ```no_run
//! use tamil_tts::mlx_ns_tts::MlxNsTts;
//! let mut tts = MlxNsTts::from_prefix("models/tamil_ns")?;
//! tts.save("வணக்கம்", "hello.wav", 1.0)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod mlx_ns_tts;
pub mod normalize;
