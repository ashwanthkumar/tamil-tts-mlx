//! tamil-tts Rust SDK: load the ONNX VITS model and synthesize Tamil speech on CPU.
//!
//! ```no_run
//! use tamil_tts::{TamilTts, SynthesisOptions};
//! let tts = TamilTts::from_file("models/tamil_female.onnx")?;
//! tts.save("வணக்கம்", "hello.wav", &SynthesisOptions::default())?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod phonemize;
pub mod mlx_tts;
pub mod mlx_ns_tts;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;

pub use phonemize::Tokenizer;

/// VITS inference knobs, mapped to the model's `scales` input.
#[derive(Debug, Clone, Copy)]
pub struct SynthesisOptions {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_scale_w: f32,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_scale_w: 0.8,
        }
    }
}

pub struct TamilTts {
    session: Session,
    tokenizer: Tokenizer,
}

impl TamilTts {
    /// Load a model. The tokenizer json is expected alongside it as `<model>.tokenizer.json`.
    pub fn from_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tok_path = tokenizer_path_for(model_path);
        Self::with_tokenizer(model_path, tok_path)
    }

    pub fn with_tokenizer<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        tokenizer_path: Q,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let session = Session::builder()
            .context("creating ORT session builder")?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("loading model {}", model_path.as_ref().display()))?;
        Ok(Self { session, tokenizer })
    }

    pub fn sample_rate(&self) -> u32 {
        self.tokenizer.sample_rate
    }

    /// Synthesize `text` to a mono f32 waveform in [-1, 1] at `self.sample_rate()`.
    pub fn synthesize(&mut self, text: &str, opts: &SynthesisOptions) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(text)?;
        let len = ids.len();

        let input = Tensor::from_array(([1_usize, len], ids))?;
        let input_lengths = Tensor::from_array(([1_usize], vec![len as i64]))?;
        let scales = Tensor::from_array((
            [3_usize],
            vec![opts.noise_scale, opts.length_scale, opts.noise_scale_w],
        ))?;

        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "input_lengths" => input_lengths,
            "scales" => scales,
        ])?;

        let (_shape, data) = outputs["output"].try_extract_tensor::<f32>()?;
        let mut wav: Vec<f32> = data.to_vec();

        // Guard against clipping from high noise scales.
        let peak = wav.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > 1.0 {
            for v in wav.iter_mut() {
                *v /= peak;
            }
        }
        Ok(wav)
    }

    /// Synthesize and write a 16-bit PCM WAV file.
    pub fn save<P: AsRef<Path>>(
        &mut self,
        text: &str,
        out_path: P,
        opts: &SynthesisOptions,
    ) -> Result<()> {
        let wav = self.synthesize(text, opts)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(out_path.as_ref(), spec)
            .with_context(|| format!("creating {}", out_path.as_ref().display()))?;
        for sample in wav {
            let clamped = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(clamped)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

fn tokenizer_path_for(model_path: &Path) -> PathBuf {
    // models/tamil_female.onnx -> models/tamil_female.tokenizer.json
    model_path.with_extension("tokenizer.json")
}
