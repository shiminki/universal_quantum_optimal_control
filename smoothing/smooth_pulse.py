"""
smooth_pulse.py — Low-pass filter for quantum control pulse sequences.

Strategy:
  1. The pulse has non-uniform time segments, so we cannot directly FFT the
     segment-indexed amplitudes.  Instead we:
       a) Build a piecewise-constant waveform on a fine uniform time grid.
       b) Apply an FFT brick-wall low-pass filter at `cutoff_MHz`.
       c) Re-sample at the *midpoints* of each original segment to recover
          the smoothed amplitudes while keeping the same segment structure.
  2. Clip amplitudes back to |Omega| <= Omega_max after filtering.

Usage:
    python smooth_pulse.py --input pulse_params.csv \
                           --output smoothed_pulse.csv \
                           --cutoff 300          # MHz (default 300)
                           --omega_max 80        # 2pi MHz (default 80)
                           --oversample 8        # uniform-grid oversample factor
"""

import argparse
import numpy as np
import pandas as pd


def lowpass_filter(signal, dt_us, cutoff_MHz):
    """
    FFT brick-wall low-pass filter.

    Parameters
    ----------
    signal    : 1-D real array
    dt_us     : uniform grid spacing [us]
    cutoff_MHz: cutoff frequency [MHz]  (frequencies above this are zeroed)

    Returns
    -------
    Filtered real array (same length).
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=dt_us)   # frequencies in MHz
    spectrum = np.fft.rfft(signal)
    spectrum[freqs > cutoff_MHz] = 0.0
    return np.fft.irfft(spectrum, n=N)


def smooth_pulse(input_csv: str,
                 output_csv: str,
                 cutoff_MHz: float = 300.0,
                 omega_max: float = 80.0,
                 oversample: int = 8):
    """
    Smooth a constant-amplitude pulse by filtering its phase angle φ = atan2(Oy, Ox).

    This is the correct approach for constant-amplitude (CA) pulses where |Omega|
    is fixed at omega_max throughout.  Filtering Ox and Oy independently would
    break the CA constraint.  Instead we:
      1. Compute φ[j] = atan2(Omega_y[j], Omega_x[j])
      2. Unwrap φ to remove 2π jumps.
      3. Build a piecewise-constant φ(t) on a fine uniform time grid.
      4. Apply FFT LPF to the unwrapped phase.
      5. Resample at original segment midpoints.
      6. Reconstruct Omega_x = omega_max * cos(φ), Omega_y = omega_max * sin(φ).

    If the pulse is NOT constant-amplitude, we fall back to filtering Ox/Oy
    independently and then clipping to omega_max.
    """
    df = pd.read_csv(input_csv)
    t_col  = df.columns[0]
    ox_col = df.columns[1]
    oy_col = df.columns[2]

    t_segs  = df[t_col].values.astype(float)
    omega_x = df[ox_col].values.astype(float)
    omega_y = df[oy_col].values.astype(float)

    # ── Detect constant-amplitude pulse ─────────────────────────────────────
    amplitude = np.sqrt(omega_x**2 + omega_y**2)
    amp_tv    = np.abs(np.diff(amplitude)).sum()
    is_CA     = amp_tv < 1e-3 * amplitude.mean() * len(amplitude)
    print(f"Pulse type: {'CONSTANT-AMPLITUDE (phase-only)' if is_CA else 'variable amplitude'}")
    print(f"  |Omega| range: {amplitude.min():.3f} – {amplitude.max():.3f}  (2pi MHz)")

    # ── Unwrapped phase (for CA pulse) ───────────────────────────────────────
    if is_CA:
        phase = np.unwrap(np.arctan2(omega_y, omega_x))
        signal_to_filter = phase
        amp_to_use = amplitude.mean()
    else:
        # Fall back: filter Ox and Oy independently
        amp_to_use = None
        signal_to_filter = None   # handled separately below

    # ── Build uniform-grid waveform ──────────────────────────────────────────
    T_total = t_segs.sum()
    dt_u    = t_segs.mean() / oversample
    M       = int(np.ceil(T_total / dt_u))
    t_edges = np.concatenate([[0.0], np.cumsum(t_segs)])

    def fill_uniform(values):
        arr = np.zeros(M)
        for j in range(len(t_segs)):
            i0 = min(int(round(t_edges[j]     / dt_u)), M)
            i1 = min(int(round(t_edges[j + 1] / dt_u)), M)
            if i1 > i0:
                arr[i0:i1] = values[j]
        return arr

    sample_rate_MHz = 1.0 / dt_u
    nyquist_MHz     = sample_rate_MHz / 2.0
    cutoff_eff      = min(cutoff_MHz, nyquist_MHz * 0.99)
    if cutoff_eff < cutoff_MHz:
        print(f"WARNING: cutoff clamped to Nyquist {nyquist_MHz:.1f} MHz")
    print(f"Uniform grid: {M} pts,  dt={dt_u*1e3:.4f} ns")
    print(f"Sample rate: {sample_rate_MHz:.1f} MHz,  Nyquist: {nyquist_MHz:.1f} MHz")
    print(f"LPF cutoff: {cutoff_eff:.1f} MHz")

    # ── Filter ───────────────────────────────────────────────────────────────
    def segment_midpoint_sample(arr_u):
        t_mids = 0.5 * (t_edges[:-1] + t_edges[1:])
        idx = np.clip(np.round(t_mids / dt_u).astype(int), 0, len(arr_u) - 1)
        return arr_u[idx]

    if is_CA:
        phase_u    = fill_uniform(phase)
        phase_filt = lowpass_filter(phase_u, dt_u, cutoff_eff)
        phase_sm   = segment_midpoint_sample(phase_filt)
        ox_smooth  = amp_to_use * np.cos(phase_sm)
        oy_smooth  = amp_to_use * np.sin(phase_sm)
    else:
        ox_u   = fill_uniform(omega_x)
        oy_u   = fill_uniform(omega_y)
        ox_f   = lowpass_filter(ox_u, dt_u, cutoff_eff)
        oy_f   = lowpass_filter(oy_u, dt_u, cutoff_eff)
        ox_smooth = segment_midpoint_sample(ox_f)
        oy_smooth = segment_midpoint_sample(oy_f)
        amp_sm = np.sqrt(ox_smooth**2 + oy_smooth**2)
        exceed = amp_sm > omega_max
        if exceed.any():
            sc = np.where(exceed, omega_max / amp_sm, 1.0)
            ox_smooth *= sc
            oy_smooth *= sc
            print(f"Clipped {exceed.sum()} segments to |Omega| <= {omega_max}")

    # ── Write output CSV ─────────────────────────────────────────────────────
    out_df = df.copy()
    out_df[ox_col] = ox_smooth
    out_df[oy_col] = oy_smooth
    out_df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    tv_orig   = np.abs(np.diff(omega_x)).sum() + np.abs(np.diff(omega_y)).sum()
    tv_smooth = np.abs(np.diff(ox_smooth)).sum() + np.abs(np.diff(oy_smooth)).sum()
    print(f"\nTotal Variation (lower = smoother):")
    print(f"  Original : {tv_orig:.1f}  (2pi MHz)")
    print(f"  Smoothed : {tv_smooth:.1f}  (2pi MHz)  [{100*(1-tv_smooth/tv_orig):.1f}% reduction]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-pass filter a quantum pulse sequence.")
    parser.add_argument("--input",      default="smoothing/X_pi_pulse.csv")
    parser.add_argument("--output",     default="smoothing/smoothed_pulse.csv")
    parser.add_argument("--cutoff",     type=float, default=3000.0,
                        help="LPF cutoff in MHz (default 3000)")
    parser.add_argument("--omega_max",  type=float, default=80.0,
                        help="Max |Omega| in 2pi MHz (default 80)")
    parser.add_argument("--oversample", type=int,   default=8,
                        help="Uniform grid oversample factor (default 8)")
    args = parser.parse_args()

    smooth_pulse(args.input, args.output, args.cutoff, args.omega_max, args.oversample)
