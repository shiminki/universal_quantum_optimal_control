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


# Build H_x(t) and H_y(t) from parameters and frequencies
def build_Hx_Hy(params, t, omegas_x, omegas_y):
    # params: [a1, ..., aN, alpha1_x, ..., alphaN_x, b1, ..., bN, alpha1_y, ..., alphaN_y]
    N = len(omegas_x)
    a = params[0:N]
    alpha_x = params[N:2*N]
    b = params[2*N:3*N]
    alpha_y = params[3*N:4*N]
    H_x = np.zeros_like(t)
    H_y = np.zeros_like(t)
    for n in range(N):
        H_x += a[n] * np.cos(omegas_x[n]*t + alpha_x[n])
        H_y += b[n] * np.cos(omegas_y[n]*t + alpha_y[n])
    return H_x, H_y

# Compute unitary evolution for one error sample
def propagate(H_x, H_y, t, delta, eps, X, Y, Z):
    U = np.eye(2, dtype=complex)
    dt = t[1] - t[0]
    for hx, hy in zip(H_x, H_y):
        # Enforce amplitude constraint: |H_x|^2 + |H_y|^2 <= 1/4
        norm = np.sqrt(hx**2 + hy**2)
        if norm > 0.5:
            hx = hx * 0.5 / norm
            hy = hy * 0.5 / norm
        Hc = hx * X + hy * Y
        H = (Hc + delta*Z) * (1 + eps)
        U = expm(-1j * H * dt) @ U
    return U

# Average fidelity over error samples
def average_infidelity(params, t, omegas_x, omegas_y, U_target, deltas, epss, X, Y, Z):
    # Build control amplitudes
    H_x, H_y = build_Hx_Hy(params, t, omegas_x, omegas_y)
    # Propagate for all samples
    Us = np.stack([
        propagate(H_x, H_y, t, d, e, X, Y, Z)
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
        # new basis frequencies for x and y
        omegas_x = random_frequencies(N_modes, w_min, w_max, seed and seed+rnd)
        omegas_y = random_frequencies(N_modes, w_min, w_max, seed and seed+rnd+1000)
        # initial params: a_n, alpha_x, b_n, alpha_y (all small random)
        x0 = np.zeros(4*N_modes)
        x0[:2*N_modes] = 0.1 * np.random.randn(2*N_modes)  # a_n, alpha_x
        x0[2*N_modes:] = 0.1 * np.random.randn(2*N_modes)  # b_n, alpha_y

        # wrapper for optimizer
        obj = lambda p: average_infidelity(p, t, omegas_x, omegas_y, U_target, deltas, epss, X, Y, Z)
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
            best_params = (res.x.copy(), omegas_x.copy(), omegas_y.copy())

    return best_params, best_fid

if __name__ == '__main__':
    # Example: target X-rotation by pi/2
    X, Y, Z = pauli_matrices()
    U_target = expm(-1j * (np.pi/2) * X / 2)

    N = 10

    params, fid = dcrab_optimize(U_target,
                                 T=12.0,
                                 dt=0.01,
                                 N_modes=N,
                                 rounds=3,
                                 samples=200,
                                 w_min=0,
                                 w_max=2*N * np.pi,
                                 seed=42)
    print(f"Best fidelity: {fid:.6f}")
    # params is a tuple (optimized params array, omegas_x array, omegas_y array)

    # Save parameters and frequencies to file
    best_params_array, best_omegas_x, best_omegas_y = params
    np.savez('dcrab_best_params.npz', params=best_params_array, omegas_x=best_omegas_x, omegas_y=best_omegas_y)
    print("Saved best parameters to 'dcrab_best_params.npz'.")
