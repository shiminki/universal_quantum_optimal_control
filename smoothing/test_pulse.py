"""
test_pulse.py — Evaluate gate fidelity and smoothness for original vs. smoothed pulses.

Gate model:
    V_j = exp(-i/2 * (Omega_x[j]*sx + Omega_y[j]*sy + delta*sz) * t[j])
    U_out = V_L * ... * V_1          (right-to-left time ordering)
    F(U) = (|Tr(U† X)|² + dim²) / (dim*(dim+1))   with dim=2 → (|Tr|² + 4) / 6

Disorder model:
    delta ~ Omega_max * N(0,1)  in units of 2pi MHz

Usage:
    python test_pulse.py [original.csv] [smoothed.csv] [--n_samples 2000]
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch


# ── Pauli matrices ────────────────────────────────────────────────────────────
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
X  = sx.copy()


def expm_2x2_pauli(nx, ny, nz, theta):
    """
    Compute exp(-i * theta/2 * (nx*sx + ny*sy + nz*sz)) efficiently.
    Uses the identity: exp(-i θ/2 n̂·σ) = cos(θ/2)I - i sin(θ/2) n̂·σ
    where θ = theta * |n̂|.
    """
    n = np.sqrt(nx**2 + ny**2 + nz**2)
    if n < 1e-15:
        return np.eye(2, dtype=complex)
    half_angle = 0.5 * theta * n
    c = np.cos(half_angle)
    s = np.sin(half_angle) / n
    return c * np.eye(2, dtype=complex) - 1j * s * (nx*sx + ny*sy + nz*sz)


def compute_propagator(t_segs, omega_x, omega_y, delta):
    """
    Compute U_out = V_L * ... * V_1 for a single delta value.
    Units: t in us, Omega in 2pi MHz  →  angle = Omega * t * 2pi is in radians
    BUT: since Omega is already in "2pi MHz" units and t in us,
         Omega[MHz] * t[us] is dimensionless (cycles), so angle = 2pi * Omega * t.
    
    Wait — let's be careful.  The Hamiltonian is:
        H = 1/2 * (Omega_x * sx + Omega_y * sy + delta * sz)
    with Omega in 2pi*MHz and t in us.  The exponent argument is -i * H * t, so:
        exponent = -i/2 * (Omega_x*sx + Omega_y*sy + delta*sz) * t
    where Omega * t has units (2pi MHz)(us) = 2pi * (MHz * us) = 2pi * (dimensionless).
    
    Since the user wrote V_j = exp(-1j * 1/2 * (...) * t[j]), the angle for the
    Pauli exponential is: theta = t[j]  (the magnitude factor goes into nx,ny,nz).
    Using our helper: exp(-i theta/2 * n̂·σ) with theta = |vec| * t[j]:
        |vec| = sqrt(Omega_x^2 + Omega_y^2 + delta^2)
        angle  = |vec| * t[j]   [in radians, since 2pi*MHz * us = 2pi]
    """
    U = np.eye(2, dtype=complex)
    for t, ox, oy in zip(t_segs, omega_x, omega_y):
        # Omega [2pi MHz] * t [us] = Omega * t * 2pi (in radians),
        # so the exponent angle needs an extra 2pi factor.
        V = expm_2x2_pauli(ox, oy, delta, 2 * np.pi * t)
        U = V @ U
    return U


def gate_fidelity(U, target=X):
    """F(U) = (|Tr(U† target)|² + 2) / 6  [dim=2 average gate fidelity]."""
    overlap = np.trace(U.conj().T @ target)
    return (abs(overlap)**2 + 2) / 6


def average_fidelity(t_segs, omega_x, omega_y,
                     omega_max=80.0, n_samples=2000, seed=42):
    """Monte-Carlo average fidelity over delta ~ omega_max * N(0,1)."""
    rng = np.random.default_rng(seed)
    deltas = omega_max * rng.standard_normal(n_samples)
    fids = []
    for delta in deltas:
        U = compute_propagator(t_segs, omega_x, omega_y, delta)
        fids.append(gate_fidelity(U))
    return np.mean(fids), np.std(fids)


# ── Smoothness metrics ────────────────────────────────────────────────────────

def total_variation(arr):
    return float(np.abs(np.diff(arr)).sum())


def max_jump(arr):
    return float(np.abs(np.diff(arr)).max())


def rms_second_derivative(arr):
    d2 = np.diff(arr, n=2)
    return float(np.sqrt(np.mean(d2**2)))


def power_above_cutoff(arr, dt_us, cutoff_MHz):
    """Fraction of signal power above cutoff_MHz."""
    freqs = np.fft.rfftfreq(len(arr), d=dt_us)
    spectrum = np.abs(np.fft.rfft(arr))**2
    total = spectrum.sum()
    if total < 1e-30:
        return 0.0
    above = spectrum[freqs > cutoff_MHz].sum()
    return float(above / total)


def print_smoothness_report(label, omega_x, omega_y, dt_us, cutoff_MHz):
    print(f"\n  [{label}]")
    combined = np.sqrt(omega_x**2 + omega_y**2)
    for name, arr in [("Omega_x", omega_x), ("Omega_y", omega_y), ("|Omega|", combined)]:
        tv  = total_variation(arr)
        mj  = max_jump(arr)
        rms = rms_second_derivative(arr)
        pwr = power_above_cutoff(arr, dt_us, cutoff_MHz)
        print(f"    {name:<10}  TV={tv:8.2f}  MaxJump={mj:6.2f}  "
              f"RMS_d2={rms:6.3f}  HF_power={pwr:.4f}")


def load_pulse(csv_path):
    df = pd.read_csv(csv_path)
    t  = df.iloc[:, 0].values.astype(float)
    ox = df.iloc[:, 1].values.astype(float)
    oy = df.iloc[:, 2].values.astype(float)
    return t, ox, oy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original",  nargs="?", default="smoothing/X_pi_pulse.csv")
    parser.add_argument("smoothed",  nargs="?", default="smoothing/smoothed_pulse.csv")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Monte-Carlo samples for fidelity (default 2000)")
    parser.add_argument("--omega_max", type=float, default=80.0)
    parser.add_argument("--cutoff",    type=float, default=300.0,
                        help="Reference cutoff MHz for HF-power metric")
    args = parser.parse_args()

    t_orig, ox_orig, oy_orig = load_pulse(args.original)
    t_sm,   ox_sm,   oy_sm   = load_pulse(args.smoothed)

    assert np.allclose(t_orig, t_sm), \
        "Segment durations differ between original and smoothed — check output."

    dt_mean = t_orig.mean()   # us (for spectral metrics, approximate uniform spacing)

    print("=" * 60)
    print("  GATE FIDELITY  (Monte-Carlo, N={})".format(args.n_samples))
    print("=" * 60)

    fid_orig, std_orig = average_fidelity(t_orig, ox_orig, oy_orig,
                                          args.omega_max, args.n_samples)
    fid_sm,   std_sm   = average_fidelity(t_sm,   ox_sm,   oy_sm,
                                          args.omega_max, args.n_samples)

    print(f"\n  Original  : {100*fid_orig:.3f}% ± {100*std_orig:.3f}%")
    print(f"  Smoothed  : {100*fid_sm:.3f}%  ± {100*std_sm:.3f}%")
    delta_fid = fid_sm - fid_orig
    print(f"  Δ Fidelity: {100*delta_fid:+.3f}%")

    PASS_THRESH = 0.99
    status = "✅ PASS (≥ 99%)" if fid_sm >= PASS_THRESH else "❌ FAIL (< 99%)"
    print(f"\n  Smoothed fidelity threshold (99%): {status}")

    print("\n" + "=" * 60)
    print("  SMOOTHNESS METRICS")
    print("=" * 60)
    print("  (TV = total variation, MaxJump = max segment-to-segment jump,")
    print("   RMS_d2 = RMS of 2nd differences, HF_power = power above cutoff)")

    print_smoothness_report("Original", ox_orig, oy_orig, dt_mean, args.cutoff)
    print_smoothness_report("Smoothed", ox_sm,   oy_sm,   dt_mean, args.cutoff)

    # Ratio summary
    tv_o = total_variation(ox_orig) + total_variation(oy_orig)
    tv_s = total_variation(ox_sm)   + total_variation(oy_sm)
    print(f"\n  Combined TV reduction: {100*(1 - tv_s/tv_o):.1f}%")

    print("\n" + "=" * 60)
    print("  DELTA=0 SANITY CHECK (no disorder)")
    print("=" * 60)
    U_orig_0 = compute_propagator(t_orig, ox_orig, oy_orig, 0.0)
    U_sm_0   = compute_propagator(t_sm,   ox_sm,   oy_sm,   0.0)
    print(f"  Original  F(delta=0): {100*gate_fidelity(U_orig_0):.4f}%")
    print(f"  Smoothed  F(delta=0): {100*gate_fidelity(U_sm_0):.4f}%")

    # Check how close to X gate
    print(f"  Original  |Tr(U†X)|:  {abs(np.trace(U_orig_0.conj().T @ X)):.6f} (max=2)")
    print(f"  Smoothed  |Tr(U†X)|:  {abs(np.trace(U_sm_0.conj().T @ X)):.6f} (max=2)")
    print()


if __name__ == "__main__":
    main()
