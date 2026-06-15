//! Rust SDK for the non-AR FastTTS exported to ONNX (two graphs, single forward, no AR loop).
//!
//! enc_dur.onnx: tokens -> (enc, log_dur); host length-regulates; decoder.onnx: (enc, expand_idx) -> mel.
//! Then Griffin-Lim (rustfft) -> wav. Artifacts from `tamiltts.mlx.export_onnx_ns`:
//!   <prefix>.enc_dur.onnx, <prefix>.decoder.onnx, <prefix>.tokenizer.json
//!
//! ```no_run
//! use tamil_tts::mlx_ns_tts::MlxNsTts;
//! let mut tts = MlxNsTts::from_prefix("models/tamil_ns")?;
//! tts.save("வணக்கம்", "hello.wav", 1.0)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;

const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;

#[derive(Debug, Deserialize)]
struct AudioCfg { sr: u32, n_fft: usize, hop: usize, win: usize, n_mels: usize }

#[derive(Debug, Deserialize)]
struct Meta {
    vocab: HashMap<String, i64>,
    mel_mean: Vec<f32>,
    mel_std: Vec<f32>,
    audio: AudioCfg,
    mel_inv: Vec<Vec<f32>>,
}

pub struct MlxNsTts {
    enc: Session,
    dec: Session,
    voc: Option<Session>,   // HiFi-GAN mel->wav; None => Griffin-Lim fallback
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
        // optional neural vocoder: hifigan.onnx in the same dir as the model prefix
        let voc_path = Path::new(p).parent().unwrap_or(Path::new("."))
            .join("hifigan.onnx");
        let voc = if voc_path.exists() {
            Some(Session::builder()?.commit_from_file(&voc_path)
                .with_context(|| format!("loading {}", voc_path.display()))?)
        } else { None };
        Ok(Self { enc, dec, voc, meta })
    }

    pub fn sample_rate(&self) -> u32 { self.meta.audio.sr }

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = vec![BOS_ID];
        for ch in text.chars() {
            if let Some(&id) = self.meta.vocab.get(&ch.to_string()) { ids.push(id); }
        }
        ids.push(EOS_ID);
        ids
    }

    pub fn synthesize(&mut self, text: &str, speed: f32) -> Result<Vec<f32>> {
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

        // decoder graph
        let enc_t = Tensor::from_array(([1usize, tt, d], enc_vec))?;
        let idx_t = Tensor::from_array(([1usize, tm], expand))?;
        let dout = self.dec.run(ort::inputs!["enc" => enc_t, "expand_idx" => idx_t])?;
        let (_ms, mel) = dout["mel_post"].try_extract_tensor::<f32>()?;
        let n_mels = self.meta.audio.n_mels;

        // denormalize log-mel
        let mut logmel = mel.to_vec();
        for (i, v) in logmel.iter_mut().enumerate() {
            let c = i % n_mels;
            *v = *v * self.meta.mel_std[c] + self.meta.mel_mean[c];
        }

        let mut wav = if self.voc.is_some() {
            // HiFi-GAN: mel as (1, n_mels, T) channel-major
            let mut mel_cht = vec![0.0f32; n_mels * tm];
            for t in 0..tm {
                for c in 0..n_mels { mel_cht[c * tm + t] = logmel[t * n_mels + c]; }
            }
            let mel_t = Tensor::from_array(([1usize, n_mels, tm], mel_cht))?;
            let voc = self.voc.as_mut().unwrap();
            let out = voc.run(ort::inputs!["mel" => mel_t])?;
            let (_s, w) = out["wav"].try_extract_tensor::<f32>()?;
            w.to_vec()
        } else {
            // Griffin-Lim fallback: mel -> linear magnitude (F x T) -> GL
            let n_fft = self.meta.audio.n_fft;
            let f_bins = n_fft / 2 + 1;
            let mut lin = vec![0.0f32; f_bins * tm];
            for t in 0..tm {
                for c in 0..n_mels {
                    let m = logmel[t * n_mels + c].exp();
                    if m == 0.0 { continue; }
                    for f in 0..f_bins { lin[f * tm + t] += self.meta.mel_inv[f][c] * m; }
                }
            }
            for v in lin.iter_mut() { if *v < 0.0 { *v = 0.0; } }
            griffin_lim(&lin, f_bins, tm, n_fft, self.meta.audio.hop, self.meta.audio.win, 60)
        };

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

fn hann(win: usize) -> Vec<f32> {
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

fn griffin_lim(mag: &[f32], f_bins: usize, n_frames: usize, n_fft: usize, hop: usize, win: usize, iters: usize) -> Vec<f32> {
    let window = hann(win);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);
    let mut spec: Vec<Complex<f32>> = (0..f_bins * n_frames).map(|i| Complex::new(mag[i], 0.0)).collect();
    let mut wav = vec![0.0f32; (n_frames - 1) * hop + win];
    for _ in 0..iters {
        wav = istft(&spec, f_bins, n_frames, n_fft, hop, win, &window, &*ifft);
        let ns = stft(&wav, n_frames, n_fft, hop, win, &window, &*fft);
        for i in 0..spec.len() {
            let p = ns[i]; let n = (p.re * p.re + p.im * p.im).sqrt();
            spec[i] = if n > 1e-8 { Complex::new(mag[i] * p.re / n, mag[i] * p.im / n) } else { Complex::new(mag[i], 0.0) };
        }
    }
    istft(&spec, f_bins, n_frames, n_fft, hop, win, &window, &*ifft)
}

fn stft(x: &[f32], n_frames: usize, n_fft: usize, hop: usize, win: usize, window: &[f32], fft: &dyn rustfft::Fft<f32>) -> Vec<Complex<f32>> {
    let f_bins = n_fft / 2 + 1;
    let mut out = vec![Complex::new(0.0, 0.0); f_bins * n_frames];
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    for t in 0..n_frames {
        let start = t * hop;
        for i in 0..n_fft {
            let s = if i < win && start + i < x.len() { x[start + i] * window[i] } else { 0.0 };
            buf[i] = Complex::new(s, 0.0);
        }
        fft.process(&mut buf);
        for f in 0..f_bins { out[f * n_frames + t] = buf[f]; }
    }
    out
}

fn istft(spec: &[Complex<f32>], f_bins: usize, n_frames: usize, n_fft: usize, hop: usize, win: usize, window: &[f32], ifft: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let len = (n_frames - 1) * hop + win;
    let mut wav = vec![0.0f32; len];
    let mut wsum = vec![0.0f32; len];
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    for t in 0..n_frames {
        for f in 0..f_bins { buf[f] = spec[f * n_frames + t]; }
        for f in 1..(n_fft - f_bins + 1) { buf[f_bins - 1 + f] = spec[(f_bins - 1 - f) * n_frames + t].conj(); }
        ifft.process(&mut buf);
        let start = t * hop;
        for i in 0..win {
            if start + i < len {
                wav[start + i] += buf[i].re / n_fft as f32 * window[i];
                wsum[start + i] += window[i] * window[i];
            }
        }
    }
    // normalize by window-overlap; zero under-overlapped edges (else tiny wsum -> spikes)
    let max_ws = wsum.iter().fold(0.0f32, |m, &v| m.max(v));
    let floor = max_ws * 1e-2;
    for i in 0..len {
        if wsum[i] > floor { wav[i] /= wsum[i]; } else { wav[i] = 0.0; }
    }
    wav
}
