"""PCA analysis of transformer control pulses.

Explores the low-dimensional structure of phi(t; theta, alpha) —
the control sequence that implements U(n(theta, phi=0), alpha).

Usage:
    cd universal_quantum_optimal_control
    python -m pca.run_pca --model_size 400 --N_theta 40 --N_alpha 50 --out_dir pca/results
"""
