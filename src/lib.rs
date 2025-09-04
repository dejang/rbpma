#![feature(error_generic_member_access)]
mod error;
use std::{error::Error, io::Cursor};

use rustfft::{FftPlanner, num_complex::Complex32};
use symphonia::{
    core::{
        audio::{AudioBuffer, Signal},
        codecs::{CODEC_TYPE_NULL, DecoderOptions},
        errors::Error as SymphoniaError,
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    },
    default::get_probe,
};

use crate::error::RbpmaError;

pub fn decode_bpm(
    input: &[u8],
    min_bpm: f32,
    max_bpm: f32,
    window_size: usize,
    hop: usize,
) -> Result<f32, Box<dyn Error>> {
    let (buffer, sample_rate) = decode_to_mono(input)?;

    if buffer.len() < window_size {
        return Err(Box::new(RbpmaError::new(
            "audio too short for chosen window size",
        )));
    }

    // 1) onset envelope via spectral flux
    let onset_env = spectral_flux(&buffer, sample_rate, window_size, hop);
    // 2) tempo from autocorrelation of onset envelope
    let bpm = estimate_bpm_from_onset(&onset_env, sample_rate, hop, min_bpm, max_bpm);

    Ok(bpm)
}

fn decode_to_mono(src: &[u8]) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let src = Cursor::new(src.to_vec());
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let hint = Hint::new();
    let probed = get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| Box::new(RbpmaError::new("no supported audio tracks")))?;

    let dec_opts: DecoderOptions = Default::default();

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;
    let track_id = track.id;
    let mut mono_buffer: Vec<f32> = vec![];
    let mut sample_rate: u32 = 44000;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::ResetRequired) => break,
            Err(SymphoniaError::IoError(error)) => match error.kind() {
                std::io::ErrorKind::UnexpectedEof => break,
                _ => return Err(Box::new(error)),
            },
            Err(error) => return Err(Box::new(error)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let signal_spec = decoded.spec();
                sample_rate = signal_spec.rate;
                let mut buf_f32: AudioBuffer<f32> = decoded.make_equivalent();
                decoded.convert(&mut buf_f32);

                let channels = signal_spec.channels.count();
                if channels == 1 {
                    mono_buffer.extend_from_slice(buf_f32.chan(0));
                } else {
                    // downcast to mono by averaging channels
                    let frames = buf_f32.frames();
                    let mut tmp = vec![0f32; frames];
                    for ch in 0..channels {
                        let c = buf_f32.chan(ch);
                        for (i, s) in c.iter().enumerate() {
                            tmp[i] += *s;
                        }
                    }
                    for s in &mut tmp {
                        *s /= channels as f32;
                    }
                    mono_buffer.extend_from_slice(&tmp);
                }
            }
            Err(SymphoniaError::IoError(_)) => {
                // The packet failed to decode due to an IO error, skip the packet.
                continue;
            }
            Err(SymphoniaError::DecodeError(_)) => {
                // The packet failed to decode due to invalid data, skip the packet.
                continue;
            }
            Err(err) => {
                // An unrecoverable error occured, halt decoding.
                return Err(Box::new(err));
            }
        }
    }
    Ok((mono_buffer, sample_rate))
}

/// Hann window
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::TAU * (i as f32) / (n as f32);
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// Spectral flux onset envelope (log-compressed & lightly smoothed).
fn spectral_flux(x: &[f32], sr: u32, win: usize, hop: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(win);

    let window = hann(win);
    let n_frames = 1 + (x.len().saturating_sub(win)) / hop;

    let mut prev_mag = vec![0f32; win / 2 + 1];
    let mut flux = Vec::<f32>::with_capacity(n_frames);

    let mut buf = vec![Complex32::new(0.0, 0.0); win];

    for f in 0..n_frames {
        let start = f * hop;
        // windowed frame into complex buffer
        for i in 0..win {
            let s = x[start + i] * window[i];
            buf[i].re = s;
            buf[i].im = 0.0;
        }
        fft.process(&mut buf);

        // magnitude spectrum (one-sided)
        let mut sum_pos_diff = 0.0f32;
        for k in 0..=win / 2 {
            let re = buf[k].re;
            let im = buf[k].im;
            let mag = (re * re + im * im).sqrt();
            let d = (mag - prev_mag[k]).max(0.0);
            sum_pos_diff += d;
            prev_mag[k] = mag;
        }

        // Log compression helps stabilize dynamic range
        flux.push((1.0 + sum_pos_diff).ln());
    }

    // Light smoothing (moving average over ~80–100 ms)
    let smooth_frames = ((0.09 * sr as f32) / hop as f32).clamp(1.0, 16.0) as usize;
    moving_average(&flux, smooth_frames)
}

fn moving_average(x: &[f32], w: usize) -> Vec<f32> {
    if w <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = vec![0f32; x.len()];
    let mut acc = 0f32;
    let mut q = std::collections::VecDeque::<f32>::new();

    for (i, &v) in x.iter().enumerate() {
        acc += v;
        q.push_back(v);
        if q.len() > w {
            acc -= q.pop_front().unwrap();
        }
        out[i] = acc / (q.len() as f32);
    }
    out
}

/// Tempo estimation from onset envelope via FFT-based autocorrelation.
fn estimate_bpm_from_onset(onset: &[f32], sr: u32, hop: usize, min_bpm: f32, max_bpm: f32) -> f32 {
    // Zero-mean the envelope to sharpen the ACF peak
    let mean = onset.iter().copied().sum::<f32>() / (onset.len().max(1) as f32);
    let mut env: Vec<f32> = onset.iter().map(|v| v - mean).collect();

    // Next power of two for efficient FFT
    let n = env.len().next_power_of_two();
    env.resize(n, 0.0);

    // FFT(env)
    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);

    let mut spec: Vec<Complex32> = env.iter().map(|&v| Complex32::new(v, 0.0)).collect();

    fwd.process(&mut spec);

    // Power spectrum |X|^2
    for c in &mut spec {
        *c = Complex32::new(c.re * c.re + c.im * c.im, 0.0);
    }

    // IFFT to get (unnormalized) autocorrelation
    inv.process(&mut spec);

    // Real part of ACF; ignore lag 0
    let mut acf: Vec<f32> = spec.iter().map(|c| c.re.max(0.0)).collect();
    for v in &mut acf {
        *v /= n as f32; // scale
    }
    if !acf.is_empty() {
        acf[0] = 0.0;
    }

    // Convert BPM bounds -> lag indices
    let lag_from_bpm = |bpm: f32| -> usize {
        let lag = (60.0 * sr as f32 / (hop as f32 * bpm)).round();
        lag.clamp(1.0, (acf.len() - 1) as f32) as usize
    };
    let lag_min = lag_from_bpm(max_bpm);
    let lag_max = lag_from_bpm(min_bpm).min(acf.len() - 1);

    // Octave-aware scoring: consider lag, lag/2, 2*lag
    let score = |lag: usize| -> f32 {
        let base = acf.get(lag).copied().unwrap_or(0.0);
        let half = acf.get(lag / 2).copied().unwrap_or(0.0) * 0.5;
        let dbl = acf.get(lag * 2).copied().unwrap_or(0.0) * 0.5;
        base + half + dbl
    };

    // Pick best lag
    let mut best_lag = lag_min;
    let mut best_val = f32::MIN;
    for lag in lag_min..=lag_max {
        let s = score(lag);
        if s > best_val {
            best_val = s;
            best_lag = lag;
        }
    }

    // Parabolic interpolation around peak for sub-lag refinement
    let lag_f = parabolic_peak(&acf, best_lag);
    let bpm = 60.0 * sr as f32 / (hop as f32 * lag_f);
    bpm.clamp(min_bpm, max_bpm)
}

fn parabolic_peak(y: &[f32], i: usize) -> f32 {
    if i == 0 || i + 1 >= y.len() {
        return i as f32;
    }
    let y0 = y[i - 1];
    let y1 = y[i];
    let y2 = y[i + 1];
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-12 {
        i as f32
    } else {
        // Vertex of parabola through (i-1,y0),(i,y1),(i+1,y2)
        i as f32 + 0.5 * (y0 - y2) / denom
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read};

    use super::*;

    #[test]
    fn it_works() {
        let mut file = File::open("Nesta - James Bande (Edit).aiff").unwrap();
        let mut buf: Vec<u8> = vec![];
        file.read_to_end(&mut buf).unwrap();

        println!("{:?}", decode_bpm(&buf, 70.0, 240.0, 2048, 1024));
    }
}
