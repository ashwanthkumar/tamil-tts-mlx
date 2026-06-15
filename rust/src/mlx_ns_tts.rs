//! Rust SDK for the non-AR FastTTS exported to ONNX (two graphs, single forward, no AR loop).
//!
//! enc_dur.onnx: tokens -> (enc, log_dur); host length-regulates; decoder.onnx: (enc, expand_idx) -> mel;
//! hifigan.onnx: mel -> wav. Artifacts from `tamiltts.mlx.export_onnx_ns` + `tamiltts.mlx.export_hifigan`:
//!   <prefix>.enc_dur.onnx, <prefix>.decoder.onnx, <prefix>.tokenizer.json, and hifigan.onnx alongside.
//!
//! ```no_run
//! use tamil_tts::mlx_ns_tts::MlxNsTts;
//! let mut tts = MlxNsTts::from_prefix("models/tamil_ns")?;
//! tts.save("வணக்கம்", "hello.wav", 1.0)?;   // 3rd arg = speed: >1 faster, <1 slower
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;

#[derive(Debug, Deserialize)]
struct AudioCfg { sr: u32, n_mels: usize }

#[derive(Debug, Deserialize)]
struct Meta {
    vocab: HashMap<String, i64>,
    mel_mean: Vec<f32>,
    mel_std: Vec<f32>,
    audio: AudioCfg,
}

pub struct MlxNsTts {
    enc: Session,
    dec: Session,
    voc: Session, // HiFi-GAN mel->wav (required)
    meta: Meta,
}

impl MlxNsTts {
    pub fn from_prefix<P: AsRef<str>>(prefix: P) -> Result<Self> {
        let p = prefix.as_ref();
        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(format!("{p}.tokenizer.json")).context("reading tokenizer")?,
        ).context("parsing tokenizer.json")?;
        let enc = Session::builder()?.commit_from_file(format!("{p}.enc_dur.onnx"))
            .context("loading enc_dur.onnx")?;
        let dec = Session::builder()?.commit_from_file(format!("{p}.decoder.onnx"))
            .context("loading decoder.onnx")?;
        // required HiFi-GAN vocoder: hifigan.onnx in the same dir as the model prefix
        let voc_path = Path::new(p).parent().unwrap_or(Path::new(".")).join("hifigan.onnx");
        if !voc_path.exists() {
            return Err(anyhow!(
                "HiFi-GAN vocoder not found at {} — it is required (export with \
                 `tamiltts.mlx.export_hifigan`)", voc_path.display()));
        }
        let voc = Session::builder()?.commit_from_file(&voc_path)
            .with_context(|| format!("loading {}", voc_path.display()))?;
        Ok(Self { enc, dec, voc, meta })
    }

    pub fn sample_rate(&self) -> u32 { self.meta.audio.sr }

    fn encode(&self, text: &str) -> Vec<i64> {
        let text = crate::normalize::normalize(text);   // verbalize acronyms/symbols/digits
        let mut ids = vec![BOS_ID];
        for ch in text.chars() {
            if let Some(&id) = self.meta.vocab.get(&ch.to_string()) { ids.push(id); }
        }
        ids.push(EOS_ID);
        ids
    }

    /// `speed` is a duration multiplier applied host-side: >1 shortens (faster),
    /// <1 lengthens (slower). Non-positive values fall back to 1.0 to avoid div-by-zero.
    pub fn synthesize(&mut self, text: &str, speed: f32) -> Result<Vec<f32>> {
        let speed = if speed > 0.0 { speed } else { 1.0 };
        let tokens = self.encode(text);
        let tt = tokens.len();

        // enc_dur graph
        let tok_t = Tensor::from_array(([1usize, tt], tokens.clone()))?;
        let out = self.enc.run(ort::inputs!["tokens" => tok_t])?;
        let (enc_shape, enc_data) = out["enc"].try_extract_tensor::<f32>()?;
        let (_ld_shape, log_dur) = out["log_dur"].try_extract_tensor::<f32>()?;
        let d = enc_shape[2] as usize;
        let enc_vec: Vec<f32> = enc_data.to_vec();

        // host length-regulation: dur = max(round(exp(log_dur)-1)/speed, 0); build expand_idx
        let mut expand: Vec<i64> = Vec::new();
        for (i, &ld) in log_dur.iter().enumerate().take(tt) {
            let dur = (((ld.exp() - 1.0) / speed).round()).max(0.0) as i64;
            for _ in 0..dur { expand.push(i as i64); }
        }
        if expand.is_empty() { expand = (0..tt as i64).collect(); }
        let tm = expand.len();

        // decoder graph -> mel
        let enc_t = Tensor::from_array(([1usize, tt, d], enc_vec))?;
        let idx_t = Tensor::from_array(([1usize, tm], expand))?;
        let dout = self.dec.run(ort::inputs!["enc" => enc_t, "expand_idx" => idx_t])?;
        let (_ms, mel) = dout["mel_post"].try_extract_tensor::<f32>()?;
        let n_mels = self.meta.audio.n_mels;

        // denormalize log-mel, lay out channel-major (1, n_mels, T) for HiFi-GAN
        let mut mel_cht = vec![0.0f32; n_mels * tm];
        for t in 0..tm {
            for c in 0..n_mels {
                mel_cht[c * tm + t] = mel[t * n_mels + c] * self.meta.mel_std[c] + self.meta.mel_mean[c];
            }
        }
        let mel_t = Tensor::from_array(([1usize, n_mels, tm], mel_cht))?;
        let vout = self.voc.run(ort::inputs!["mel" => mel_t])?;
        let (_s, w) = vout["wav"].try_extract_tensor::<f32>()?;
        let mut wav = w.to_vec();

        let peak = wav.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > 1e-6 { for v in wav.iter_mut() { *v *= 0.95 / peak; } }
        Ok(wav)
    }

    pub fn save<P: AsRef<Path>>(&mut self, text: &str, out: P, speed: f32) -> Result<()> {
        let wav = self.synthesize(text, speed)?;
        let spec = hound::WavSpec { channels: 1, sample_rate: self.sample_rate(),
            bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut w = hound::WavWriter::create(out.as_ref(), spec)?;
        for s in wav { w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?; }
        w.finalize()?;
        Ok(())
    }
}
