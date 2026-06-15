//! Rust SDK for the MLX-trained TransformerTTS exported to ONNX.
//!
//! Char-level Tamil tokenizer (no espeak), autoregressive decode via `ort`, and a self-contained
//! Griffin-Lim vocoder (`rustfft`). The model + `<model>.tokenizer.json` (vocab, mel stats, mel
//! pseudo-inverse, audio params) are produced by `tamiltts.mlx.export_onnx`.
//!
//! ```no_run
//! use tamil_tts::mlx_tts::MlxTts;
//! let mut tts = MlxTts::from_file("models/tamil_mlx.onnx")?;
//! tts.save("வணக்கம்", "hello.wav", 800)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;

const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;

#[derive(Debug, Deserialize)]
struct AudioCfg {
    sr: u32,
    n_fft: usize,
    hop: usize,
    win: usize,
    n_mels: usize,
}

#[derive(Debug, Deserialize)]
struct TokenizerMeta {
    vocab: HashMap<String, i64>,
    mel_mean: Vec<f32>,
    mel_std: Vec<f32>,
    audio: AudioCfg,
    mel_inv: Vec<Vec<f32>>, // (1 + n_fft/2, n_mels)
}

pub struct MlxTts {
    session: Session,
    meta: TokenizerMeta,
}

impl MlxTts {
    pub fn from_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let tok_path = tokenizer_path_for(model_path.as_ref());
        Self::with_tokenizer(model_path, tok_path)
    }

    pub fn with_tokenizer<P: AsRef<Path>, Q: AsRef<Path>>(model_path: P, tok_path: Q) -> Result<Self> {
        let data = std::fs::read_to_string(tok_path.as_ref())
            .with_context(|| format!("reading tokenizer {}", tok_path.as_ref().display()))?;
        let meta: TokenizerMeta = serde_json::from_str(&data).context("parsing tokenizer.json")?;
        let session = Session::builder()
            .context("ORT session builder")?
            .commit_from_file(model_path.as_ref())
            .with_context(|| format!("loading model {}", model_path.as_ref().display()))?;
        Ok(Self { session, meta })
    }

    pub fn sample_rate(&self) -> u32 {
        self.meta.audio.sr
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = vec![BOS_ID];
        for ch in text.chars() {
            if let Some(&id) = self.meta.vocab.get(&ch.to_string()) {
                ids.push(id);
            }
        }
        ids.push(EOS_ID);
        ids
    }

    /// Autoregressively decode a denormalized log-mel (row-major frames, each `n_mels` long).
    fn synth_mel(&mut self, text: &str, max_frames: usize) -> Result<Vec<f32>> {
        let n_mels = self.meta.audio.n_mels;
        let tokens = self.encode(text);
        let tt = tokens.len();

        // mel_in starts with a single zero "go" frame; grows by one predicted frame per step.
        let mut mel_in: Vec<f32> = vec![0.0; n_mels];
        let mut out: Vec<f32> = Vec::new();

        for _ in 0..max_frames {
            let t_now = mel_in.len() / n_mels;
            let tok_t = Tensor::from_array(([1usize, tt], tokens.clone()))?;
            let mel_t = Tensor::from_array(([1usize, t_now, n_mels], mel_in.clone()))?;
            let outputs = self
                .session
                .run(ort::inputs!["tokens" => tok_t, "mel_in" => mel_t])?;

            let (_s, mel_post) = outputs["mel_post"].try_extract_tensor::<f32>()?;
            let (_s2, stop) = outputs["stop"].try_extract_tensor::<f32>()?;

            // last frame of mel_post
            let last = &mel_post[(t_now - 1) * n_mels..t_now * n_mels];
            out.extend_from_slice(last);
            mel_in.extend_from_slice(last);

            let stop_last = stop[t_now - 1];
            if 1.0 / (1.0 + (-stop_last).exp()) > 0.5 {
                break;
            }
        }

        // denormalize: mel * std + mean
        for (i, v) in out.iter_mut().enumerate() {
            let c = i % n_mels;
            *v = *v * self.meta.mel_std[c] + self.meta.mel_mean[c];
        }
        Ok(out)
    }

    pub fn synthesize(&mut self, text: &str, max_frames: usize) -> Result<Vec<f32>> {
        let logmel = self.synth_mel(text, max_frames)?;
        let n_mels = self.meta.audio.n_mels;
        let n_frames = logmel.len() / n_mels;
        if n_frames == 0 {
            return Err(anyhow!("no frames generated"));
        }
        // mel magnitude (n_mels x T) = exp(logmel)
        // linear magnitude (F x T) = mel_inv (F x n_mels) @ mel
        let n_fft = self.meta.audio.n_fft;
        let f_bins = n_fft / 2 + 1;
        let mut lin = vec![0.0f32; f_bins * n_frames]; // row-major (F, T)
        for t in 0..n_frames {
            for c in 0..n_mels {
                let m = logmel[t * n_mels + c].exp();
                if m == 0.0 {
                    continue;
                }
                for f in 0..f_bins {
                    lin[f * n_frames + t] += self.meta.mel_inv[f][c] * m;
                }
            }
        }
        for v in lin.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        Ok(griffin_lim(&lin, f_bins, n_frames, n_fft, self.meta.audio.hop, self.meta.audio.win, 60))
    }

    pub fn save<P: AsRef<Path>>(&mut self, text: &str, out: P, max_frames: usize) -> Result<()> {
        let wav = self.synthesize(text, max_frames)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(out.as_ref(), spec)
            .with_context(|| format!("creating {}", out.as_ref().display()))?;
        for s in wav {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}

fn hann(win: usize) -> Vec<f32> {
    (0..win).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / win as f32).cos()).collect()
}

/// Griffin-Lim: reconstruct a waveform from a linear magnitude spectrogram `mag` (F x T, row-major).
/// Self-consistent STFT/ISTFT (center=False framing), Hann window, `iters` phase iterations.
fn griffin_lim(mag: &[f32], f_bins: usize, n_frames: usize, n_fft: usize, hop: usize, win: usize, iters: usize) -> Vec<f32> {
    let window = hann(win);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    // initialize complex spectrogram: magnitude with zero phase
    let mut spec: Vec<Complex<f32>> = (0..f_bins * n_frames).map(|i| Complex::new(mag[i], 0.0)).collect();

    let mut wav = vec![0.0f32; (n_frames - 1) * hop + win];
    for _ in 0..iters {
        wav = istft(&spec, f_bins, n_frames, n_fft, hop, win, &window, &*ifft);
        let new_spec = stft(&wav, n_frames, n_fft, hop, win, &window, &*fft);
        // keep magnitude, take phase from new_spec
        for i in 0..spec.len() {
            let ph = new_spec[i];
            let n = (ph.re * ph.re + ph.im * ph.im).sqrt();
            spec[i] = if n > 1e-8 { Complex::new(mag[i] * ph.re / n, mag[i] * ph.im / n) } else { Complex::new(mag[i], 0.0) };
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
        for f in 0..f_bins {
            out[f * n_frames + t] = buf[f];
        }
    }
    out
}

fn istft(spec: &[Complex<f32>], f_bins: usize, n_frames: usize, n_fft: usize, hop: usize, win: usize, window: &[f32], ifft: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let len = (n_frames - 1) * hop + win;
    let mut wav = vec![0.0f32; len];
    let mut wsum = vec![0.0f32; len];
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    for t in 0..n_frames {
        // rebuild full hermitian spectrum
        for f in 0..f_bins {
            buf[f] = spec[f * n_frames + t];
        }
        for f in 1..(n_fft - f_bins + 1) {
            buf[f_bins - 1 + f] = spec[(f_bins - 1 - f) * n_frames + t].conj();
        }
        ifft.process(&mut buf);
        let start = t * hop;
        for i in 0..win {
            if start + i < len {
                let v = buf[i].re / n_fft as f32; // rustfft inverse is unnormalized
                wav[start + i] += v * window[i];
                wsum[start + i] += window[i] * window[i];
            }
        }
    }
    for i in 0..len {
        if wsum[i] > 1e-8 {
            wav[i] /= wsum[i];
        }
    }
    wav
}

fn tokenizer_path_for(model_path: &Path) -> PathBuf {
    model_path.with_extension("tokenizer.json")
}
