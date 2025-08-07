import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

DELTA_STD = 0.4  # standard deviation of detuning error
EPSILON_STD = 0.05  # standard deviation of amplitude error


# Pauli matrices
def pauli_matrices():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return X, Y, Z

# Sample static errors
def sample_errors(n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    delta = np.random.normal(0, DELTA_STD, size=n_samples)
    eps   = np.random.normal(0, EPSILON_STD, size=n_samples)
    return delta, eps

# Build phi(t) from parameters and frequencies
def build_phi(params, t, omegas):
    # params: [phi0, a1, ..., aN, b1, ..., bN]
    N = len(omegas)
    phi0 = params[0]
    a = params[1:1+N]
    b = params[1+N:1+2*N]
    phi = phi0 + sum(a[n] * np.cos(omegas[n]*t) + b[n] * np.sin(omegas[n]*t)
                     for n in range(N))
    return phi

# Compute unitary evolution for one error sample
def propagate(phi_vals, t, delta, eps, X, Y, Z):
    U = np.eye(2, dtype=complex)
    dt = t[1] - t[0]
    for phi in phi_vals:
        Hc = np.cos(phi)*X + np.sin(phi)*Y
        H = (Hc + delta*Z) * (1 + eps) / 2
        U = expm(-1j * H * dt) @ U
    return U

# Average fidelity over error samples
def average_infidelity(params, t, omegas, U_target, deltas, epss, X, Y, Z):
    # Build control phases
    phi_vals = build_phi(params, t, omegas)
    # Propagate for all samples
    Us = np.stack([
        propagate(phi_vals, t, d, e, X, Y, Z)
        for d, e in zip(deltas, epss)
    ], axis=0)  # shape (S,2,2)
    # Apply U_target dagger and compute traces
    M = np.matmul(U_target.conj().T[np.newaxis, :, :], Us)  # (S,2,2)
    traces = np.trace(M, axis1=1, axis2=2)                # (S,)
    fidelities = (np.abs(traces)+2) / 6                        # (S,)
    return 1 - np.mean(fidelities)  # negative for minimization

# Generate random basis frequencies
def random_frequencies(N, w_min, w_max, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(w_min, w_max, size=N)

# dCRAB main routine
def dcrab_optimize(
    U_target,
    T=6.0,
    dt=0.01,
    N_modes=12,
    rounds=5,
    samples=100,
    w_min=0.0,
    w_max=10.0,
    seed=None
):
    # time grid
    t = np.arange(0, T, dt)
    X, Y, Z = pauli_matrices()

    # prepare error samples once
    deltas, epss = sample_errors(samples, seed)

    best_params = None
    best_fid = -np.inf

    print("Starting dCRAB optimization...")

    for rnd in range(rounds):
        # new basis frequencies
        omegas = random_frequencies(N_modes, w_min, w_max, seed and seed+rnd)
        # initial params: phi0=0, a_n,b_n small random
        x0 = np.zeros(1 + 2*N_modes)
        x0[1:] = 0.01 * np.random.randn(2*N_modes)

        # wrapper for optimizer
        obj = lambda p: average_infidelity(p, t, omegas, U_target, deltas, epss, X, Y, Z)
        # setup timing and iteration counter for callback
        start_time = time.time()
        iter_counter = {'i': 0}
        def callback(xk):
            iter_counter['i'] += 1
            elapsed = time.time() - start_time
            if iter_counter['i'] % 50 == 0:
                print(f"    [Round {rnd+1}] Iter {iter_counter['i']}: elapsed {elapsed:.2f}s")

        # optimize parameters with progress callback
        res = minimize(
            obj, x0,
            method='Nelder-Mead',
            callback=callback,
            options={'maxiter': 1000, 'disp': True}
        )

        # evaluate fidelity
        fid = 1 - res.fun
        print(f"Round {rnd+1}/{rounds}: fidelity = {fid:.6f}")

        if fid > best_fid:
            best_fid = fid
            best_params = (res.x.copy(), omegas.copy())

    return best_params, best_fid

if __name__ == '__main__':
    # Example: target X-rotation by pi/2
    X, Y, Z = pauli_matrices()
    U_target = expm(-1j * (np.pi/2) * X / 2)

    N = 2000

    params, fid = dcrab_optimize(U_target,
                                 T=6.0,
                                 dt=0.01,
                                 N_modes=N,
                                 rounds=5,
                                 samples=200,
                                 w_min=0.1,
                                 w_max=N * np.pi,
                                 seed=42)
    print(f"Best fidelity: {fid:.6f}")
    # params is a tuple (optimized params array, omegas array)

    # Save parameters and frequencies to file
    best_params_array, best_omegas = params
    np.savez('dcrab_best_params.npz', params=best_params_array, omegas=best_omegas)
    print("Saved best parameters to 'dcrab_best_params.npz'.")
