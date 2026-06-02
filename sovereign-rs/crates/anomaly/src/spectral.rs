//! Spectral analysis via a dependency-free radix-2 FFT — decompose a price
//! series into frequency components to surface periodic algorithmic cycles.

/// Minimal complex number for the FFT.
#[derive(Debug, Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    #[inline(always)]
    fn add(self, o: Cx) -> Cx {
        Cx {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    #[inline(always)]
    fn sub(self, o: Cx) -> Cx {
        Cx {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    #[inline(always)]
    fn mul(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p.max(1)
}

/// In-place iterative radix-2 Cooley-Tukey FFT (`buf.len()` must be a power of 2).
fn fft(buf: &mut [Cx]) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Butterflies.
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wlen = Cx {
            re: ang.cos(),
            im: ang.sin(),
        };
        let mut i = 0;
        while i < n {
            let mut w = Cx { re: 1.0, im: 0.0 };
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2].mul(w);
                buf[i + k] = u.add(v);
                buf[i + k + len / 2] = u.sub(v);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// One-sided power spectrum of a real signal (mean removed, zero-padded to the
/// next power of two). Index `k` corresponds to frequency `k / N`.
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let n0 = signal.len();
    if n0 < 2 {
        return Vec::new();
    }
    let mean = signal.iter().sum::<f64>() / n0 as f64;
    let n = next_pow2(n0);
    let mut buf: Vec<Cx> = (0..n)
        .map(|i| Cx {
            re: if i < n0 { signal[i] - mean } else { 0.0 },
            im: 0.0,
        })
        .collect();
    fft(&mut buf);
    buf[..n / 2]
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect()
}

/// Dominant cycle length (in samples) — the period of the strongest non-DC
/// frequency. Returns `0.0` if there's no clear cycle.
pub fn dominant_period(signal: &[f64]) -> f64 {
    let spec = power_spectrum(signal);
    if spec.len() < 2 {
        return 0.0;
    }
    let n = next_pow2(signal.len());
    let mut best_k = 0usize;
    let mut best_p = 0.0;
    for (k, &p) in spec.iter().enumerate().skip(1) {
        if p > best_p {
            best_p = p;
            best_k = k;
        }
    }
    if best_k == 0 {
        0.0
    } else {
        n as f64 / best_k as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_period() {
        // Pure sine with period 16 over 256 samples.
        let period = 16.0;
        let sig: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin())
            .collect();
        let p = dominant_period(&sig);
        assert!((p - period).abs() < 1.0, "got {p}");
    }

    #[test]
    fn flat_signal_has_no_cycle() {
        let sig = vec![5.0; 128];
        assert_eq!(dominant_period(&sig), 0.0);
    }
}
