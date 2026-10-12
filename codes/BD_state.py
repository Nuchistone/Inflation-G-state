#!/usr/bin/env python3
"""
BD state - Complete Analytical Solution of the Mukhanov-Sasaki (MS) Equation 
Potential: V = 1/2 m^2 phi^2

Pipeline
--------
1. Background evolution in the e-fold variable N (m^2 phi^2, slow roll) 
2. Exact BD mode functions from the Hankel solution (constant nu = nu_BD) 
3. Slow-roll observables (n_s, r, alpha_s) at the pivot and at reference 
   e-folds 
4. Power spectra and all figures (spectra, mode functions vs conformal 
   time, mode functions vs N) 

No numerical integration is needed in this file: the BD mode functions are 
analytical. The numerical integration of the MS equation lives in the 
companion G-state code, which uses this file's mode functions as its 
benchmark and initial data. 

Code organisation
-----------------
Mirrors the G-state code (G_state.py): the two files share the unit convention 
and the configuration block (Part A), and the physics kernels (Part B) use 
identical formulas, so the BD benchmark that the G-state pipeline integrates 
against is exactly the one documented here. 
Physics kernels (Part B), observables and spectra computation (Part C), 
console report printing (Part D), and figure generation (Part E) -- all 
plain function. All computation happens inside these functions and 
main() (Part F) calls the functions in order orchestrating the 
console report and the figures. 

Changing the potential
----------------------
All V(phi) dependence lives in Part A (the A_s normalisation of the scale
parameter m_phi, and the N_star / N_end conventions) and in the Part-B
kernels epsilon_of_N, eta_V_of_N (plus xi2_V_of_N in the BD code). Every
derived constant is computed from these kernels, and the ODE pipeline,
verification tests, fitting and scan are potential-independent. After a
potential swap, re-run Test 1: the atol rationale scales with H_star.

Author: Huangruixi Lyu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from pathlib import Path
import sys, contextlib


# PART A. CONFIGURATION
# Unit convention: identical to the G-state code. All internal (background) 
# quantities are in Planck units (M_pl = 1). Wavenumbers take Mpc^-1 unit 
# only through the dimensionless ratio kappa = k/k_pivot. The pipeline is 
# scale-free, so no explicit Mpc <-> M_pl conversion is needed. m_phi sets 
# only the overall amplitude and is fixed by A_s. 
N_star = 60
k_pivot = 0.05
A_s_Planck = 2.1e-9
N_end = 120
m_phi = np.pi * np.sqrt(6.0 * A_s_Planck) / (N_end - N_star)  # from A_s = m^2 (N_end - N_*)^2 / (6 pi^2)

# E-folds at which the slow-roll observables are reported on the console. 
# N = 70 is 10 e-folds after the pivot crossing (N_end - N = 50). 
OBSERVABLE_N = [70, 60, 50]

# Modes shown in the figures
K_MODES_CONFORMAL = [0.001, 0.01, 0.05, 0.1, 0.5]  # k in Mpc^-1 (conformal-time figure)
KAPPA_MODES = [1e-4, 1e-3, 1e-2, 0.1, 1.0]         # kappa = k/(aH) at N_star (e-fold figure)
KAPPA_LABELS = ['1e-4', '1e-3', '1e-2', '0.1', '1']
COLORS = ['blue', 'green', 'red', 'purple', 'orange']

# Output directory: the result/ folder next to the codes/ folder
RESULT_DIR = Path(__file__).resolve().parent.parent / "result" / "BD_state"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
FIG_SPECTRA = RESULT_DIR / 'BD_power_spectra.png'
FIG_CONFORMAL = RESULT_DIR / 'BD_conformal_time.png'
FIG_EFOLDS = RESULT_DIR / 'BD_efolds.png'

# PART B. PHYSICS KERNELS

# B1. Background
def epsilon_of_N(N):
    """Slow-roll parameter epsilon(N) = 1/(2(N_end - N)) for m^2 phi^2."""
    return 1.0 / (2.0 * (N_end - N))

def eta_V_of_N(N):
    """eta_V(N). For V ~ phi^2, eta_V = epsilon_V."""
    return epsilon_of_N(N)

def nu_of_N(N):
    """Hankel index nu(N) = 3/2 + 3 eps_V - eta_V (first-order slow roll)."""
    return 1.5 + 3.0 * epsilon_of_N(N) - eta_V_of_N(N)

def a_of_N(N):
    """Scale factor a(N) = e^N, with a = 1 at N = 0 (start of inflation)."""
    return np.exp(N)

def z_of_N(N):
    """z(N) = a(N) * sqrt(2*epsilon(N)), M_pl = 1."""
    return a_of_N(N) * np.sqrt(2.0 * epsilon_of_N(N))

def tau_of_N(N):
    """Conformal time tau(N) = -e^{-N}/H_star in the de Sitter approximation."""
    return -np.exp(-N) / H_star

epsilon_star = epsilon_of_N(N_star)
nu_BD = nu_of_N(N_star)
phi_star = 2.0 * np.sqrt(N_end - N_star)
H_star = np.sqrt(0.5 * m_phi**2 * phi_star**2 / 3.0)

# B2. BD state mode function (Hankel, constant nu)
def BD_mode(k, eta, nu):
    """
    BD state mode function v_k^BD(eta) using Hankel H_nu^(1): 
    v_k(eta) = (sqrt(pi)/2) * e^{i*pi*(2*nu+1)/4} * sqrt(-eta) * H_nu^(1)(-k*eta) 
    Inside the horizon, v_k -> e^{-ik*eta} / sqrt(2k). 
    Also accepts numpy arrays for eta, since the figures evaluate whole time 
    grids at once. 
    """
    x = np.clip(-k * eta, 1e-300, 1e300)
    C = np.sqrt(np.pi) / 2.0 * np.exp(1j * np.pi * (2.0 * nu + 1.0) / 4.0)
    return C * np.sqrt(-eta) * hankel1(nu, x)

# PART C. OBSERVABLES AND SPECTRA

def xi2_V_of_N(N):
    """
    Second potential slow-roll parameter xi_V^2 = M^4 V' V''' / V^2. 
    Identically zero for V ~ phi^2 (V''' = 0). 
    """
    return 0.0

def slow_roll_observables(N):
    """
    Slow-roll observables at e-fold N: (epsilon_V, n_s, r, alpha_s) with 
    n_s = 1 - 6 eps_V + 2 eta_V,  r = 16 eps_V, 
    alpha_s = 16 eps_V eta_V - 24 eps_V^2 - 2 xi_V^2. 
    For V ~ phi^2 (eta_V = eps_V, xi_V^2 = 0): n_s = 1 - 4 eps, alpha_s = -8 eps^2. 
    """
    eps, eta, xi2 = epsilon_of_N(N), eta_V_of_N(N), xi2_V_of_N(N)
    return (eps, 1.0 - 6.0*eps + 2.0*eta, 16.0*eps,
            16.0*eps*eta - 24.0*eps**2 - 2.0*xi2)

def P_zeta_slowroll(k, N):
    """
    Slow-roll scalar power spectrum 
    P_zeta(k) = A_s * (k/k_pivot)^{n_s(N) - 1}, 
    a local power law with the tilt evaluated at e-fold N. 
    """
    _, n_s, _, _ = slow_roll_observables(N)
    return A_s_Planck * (k / k_pivot)**(n_s - 1.0)

# PART D. CONSOLE REPORT

def print_background():
    """Background quantities at the pivot N_star."""
    print("=" * 60)
    print("BD State - MS Equation Solution")
    print("V = 1/2 * m^2 * phi^2")
    print("=" * 60)
    print(f"\n--- Background at N = {N_star} ---")
    print(f"Slow-roll parameter epsilon = {epsilon_star:.6f}")
    print(f"Slow-roll parameter eta     = {eta_V_of_N(N_star):.6f}  "
          f"(eta_V coincides with epsilon_V for V ~ phi^2)")
    print(f"Field value phi_* = {phi_star:.4f} M_P")
    print(f"Hubble parameter H_* = {H_star:.6e} M_P")
    print(f"Hankel function order nu = {nu_BD:.6f}")

def print_observables():
    """Slow-roll observables at the pivot and the reference e-folds."""
    print(f"\n--- Observables ---")
    for N_val in OBSERVABLE_N:
        _, n_s, r_val, alpha_s = slow_roll_observables(N_val)
        print(f"At N_end-N = {N_end - N_val}:")
        print(f"  n_s = {n_s:.4f}")
        print(f"  r   = {r_val:.4f}")
        print(f"  alpha_s = {alpha_s:.6f}")

def print_amplitudes():
    """Overall amplitude, inflationary energy scale and pivot spectra."""
    Delta_phi = np.sqrt(2.0 * epsilon_star) * N_star
    H_from_As = np.sqrt(A_s_Planck * 8.0 * np.pi**2 * epsilon_star)
    V04 = (3.0 * H_from_As**2)**0.25
    print(f"\nField excursion Delta_phi = {Delta_phi:.2f} M_P")
    print(f"Hubble parameter H_* (from A_s) = {H_from_As:.4e} M_P")
    print(f"Energy scale V^(1/4) = {V04:.4e} M_P")

    _, n_s, r, _ = slow_roll_observables(N_star)
    print(f"\n--- Power Spectrum at k_pivot = {k_pivot} Mpc^-1 ---")
    print(f"P_zeta(k_*) = {A_s_Planck:.4e}")
    print(f"P_t(k_*) = {r * A_s_Planck:.4e}")
    print(f"r = {r:.4f}")
    print(f"n_s = {n_s:.4f}")

# PART E. FIGURES

def make_figure_power_spectra(path=FIG_SPECTRA):
    """Slow-roll scalar spectrum, and the scalar/tensor comparison."""
    k_array = np.logspace(-4, 1, 500)
    P_zeta_k = P_zeta_slowroll(k_array, N_star)
    _, n_s, r, _ = slow_roll_observables(N_star)
    P_t_k = r * P_zeta_k

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].loglog(k_array, P_zeta_k, 'b-', lw=2, label=r'$P_\zeta(k)$')
    axes[0].axvline(k_pivot, color='gray', ls=':', alpha=0.7)
    axes[0].set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=13)
    axes[0].set_ylabel(r'$P_\zeta$', fontsize=13)
    axes[0].set_title(f'Scalar Power Spectrum, N={N_star}', fontsize=14)
    axes[0].legend(fontsize=13)
    axes[0].grid(True, which='both', ls='--', alpha=0.5)

    axes[1].loglog(k_array, P_zeta_k, 'b-', lw=2, label=r'$P_\zeta$ (scalar)')
    axes[1].loglog(k_array, P_t_k, 'r--', lw=2, label=r'$P_t$ (tensor)')
    axes[1].axvline(k_pivot, color='gray', ls=':', alpha=0.7)
    axes[1].set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=14)
    axes[1].set_ylabel('P', fontsize=14)
    axes[1].set_title(f'Scalar and Tensor, N={N_star}, $n_s$={n_s:.4f}, r={r:.4f}',
                      fontsize=15)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)

def make_figure_conformal_time(path=FIG_CONFORMAL):
    """
    |v_k| and |zeta_k|^2 versus conformal time -tau for five modes given in 
    Mpc^-1. Dashed vertical lines mark the horizon crossings -tau = 1/k. 
    """
    tau_range = np.logspace(-2, 3, 2000)  # this is -tau > 0

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: |v_k(-tau)|. Units follow from k in Mpc^-1: v_k carries Mpc^{1/2}.
    for i, k in enumerate(K_MODES_CONFORMAL):
        vk = BD_mode(k, -tau_range, nu_BD)
        axes[0].loglog(tau_range, np.abs(vk), color=COLORS[i], lw=1.5,
                       label=f'k={k}' + r'Mpc$^{-1}$')
        axes[0].axvline(1.0 / k, color=COLORS[i], ls='--', alpha=0.3)
    axes[0].set_xlabel(r'$-\tau$ [Mpc]', fontsize=14)
    axes[0].set_ylabel(r'$|v_k|$ [Mpc$^{1/2}$]', fontsize=14)
    axes[0].set_title(r'Mode Function $|v_k|$', fontsize=15)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, which='both', ls='--', alpha=0.5)

    # Bottom: |zeta_k|^2 = |v_k/z|^2. In the de Sitter approximation 
    # a = -1/(H_* tau), so z(tau) = a sqrt(2 eps) = sqrt(2 eps_*) / (H_* (-tau)). 
    z_tau = np.sqrt(2.0 * epsilon_star) / (H_star * tau_range)
    for i, k in enumerate(K_MODES_CONFORMAL):
        vk = BD_mode(k, -tau_range, nu_BD)
        zeta_k2 = (np.abs(vk) / z_tau)**2
        axes[1].loglog(tau_range, zeta_k2, color=COLORS[i], lw=1.5,
                       label=f'k={k}' + r'Mpc$^{-1}$')
        axes[1].axvline(1.0 / k, color=COLORS[i], ls='--', alpha=0.3)
    axes[1].set_xlabel(r'$-\tau$ [Mpc]', fontsize=14)
    axes[1].set_ylabel(r'$|\zeta_k|^2$ [Mpc$^3$]', fontsize=14)
    axes[1].set_title(r'Curvature Perturbation $|\zeta_k|^2$ (FREEZE-OUT)', fontsize=15)
    axes[1].legend(loc='lower right', fontsize=14)
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1].set_ylim([1e-8, 1e3])

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)

def make_figure_efolds(path=FIG_EFOLDS):
    """
    |v_k| and P_zeta(N) versus the e-fold variable for five modes labelled 
    by kappa = k/(aH) at N_star. Dashed vertical lines mark the horizon 
    crossings N_hc = N_star + ln(kappa). 
    """
    N_array = np.linspace(0.5, 65, 5000)
    tau_N = tau_of_N(N_array)
    z_N = z_of_N(N_array)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: |v_k(N)|. k is in Planck units here, so v_k carries M_pl^{-1/2}. 
    for i, kappa in enumerate(KAPPA_MODES):
        k_phys = kappa * np.exp(N_star) * H_star  # k = kappa * (aH) at N_star
        N_hc = N_star + np.log(kappa)             # horizon crossing k = aH
        vk = BD_mode(k_phys, tau_N, nu_BD)
        axes[0].semilogy(N_array, np.abs(vk), color=COLORS[i], lw=1.5,
                         label=f'k/(aH)={KAPPA_LABELS[i]}')
        if 0 < N_hc < 65:
            axes[0].axvline(N_hc, color=COLORS[i], ls='--', alpha=0.3)
    axes[0].set_xlabel('N (e-folds from the beginning of inflation)', fontsize=14)
    axes[0].set_ylabel(r'$|v_k|$ [$M_{\rm pl}^{-1/2}$]', fontsize=14)
    axes[0].set_title(r'Mode Function $|v_k|$', fontsize=15)
    axes[0].legend(loc='upper left', fontsize=14)
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    axes[0].set_xlim([0, 65])

    # Bottom: P_zeta(N) = k^3/(2 pi^2) * |v_k/z|^2, showing the freeze-out. 
    for i, kappa in enumerate(KAPPA_MODES):
        k_phys = kappa * np.exp(N_star) * H_star
        N_hc = N_star + np.log(kappa)
        vk = BD_mode(k_phys, tau_N, nu_BD)
        P_zeta_N = (k_phys**3 / (2.0 * np.pi**2)) * (np.abs(vk) / z_N)**2
        axes[1].semilogy(N_array, P_zeta_N, color=COLORS[i], lw=1.5,
                         label=f'k/(aH)={KAPPA_LABELS[i]}')
        if 0 < N_hc < 65:
            axes[1].axvline(N_hc, color=COLORS[i], ls='--', alpha=0.3)
    axes[1].set_xlabel('N (e-folds from the beginning of inflation)', fontsize=14)
    axes[1].set_ylabel(r'$P_\zeta$', fontsize=14)
    axes[1].set_title(r'Power Spectrum $P_\zeta(N)$ (FREEZE-OUT)', fontsize=15)
    axes[1].legend(loc='upper left', fontsize=14)
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1].set_xlim([0, 65])

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)

# PART F. MAIN

def main():
    print_background()
    print_observables()
    print_amplitudes()
    make_figure_power_spectra()
    make_figure_conformal_time()
    make_figure_efolds()
    print("\nAll figures saved!")
    print("BD_power_spectra.png")
    print("BD_conformal_time.png")
    print("BD_efolds.png")

class _Tee:
    """Write to the console and a log file simultaneously."""
    def __init__(self, log):
        self.log = log
    def write(self, s):
        sys.__stdout__.write(s)
        self.log.write(s)
    def flush(self):
        sys.__stdout__.flush()
        self.log.flush()

if __name__ == "__main__":
    log_path = RESULT_DIR / 'BD_console.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file, \
            contextlib.redirect_stdout(_Tee(log_file)):
        main()
