#!/usr/bin/env python3
"""
G Vacuum - Complete Numerical Solution with Parameter Scan
Potential: V = 1/2 m^2 phi^2

Pipeline
--------
1. ODE integration of the Mukhanov-Sasaki (MS) equation for the G vacuum
2. Bogoliubov-transformed initial conditions
3. P_zeta(k) computation and comparison with the BD vacuum
4. Full 5x5 parameter scan over beta_0 and k_c
5. n_s and alpha_s fitting from the power spectrum
6. Planck chi^2 exclusion analysis
7. Verification tests (see below)
8. All plots

Verification tests (printed in logical order)
---------------------------------------------
Integrator accuracy -- "error" = numerical inaccuracy of the solver itself:
    Test 1  tolerance convergence at the pivot (beta_0 = 0)
    Test 2  Wronskian conservation and normalisation
Analytical-vs-numerical deviation -- "dev" = (numerical - analytical)/analytical,
a *physical* difference between two approximation schemes, NOT a numerical
mistake. Every dev line states the quantity it refers to and its formula:
    Test 3  single mode at the pivot vs the (alpha+|beta|)^2 enhancement
    Test 4  deviation over the full k range
    Test 5  beta_0 -> 0 limit (G -> BD)
    Test 6  constant-nu experiment: identifies the slow-roll drift of nu as the
            sole origin of the dev
    Test 7  freeze-out truncation exclusion (amplitude / phase / k-independence)
    Test 8  closed-form estimate of the dev, compared against Test 6

Code organisation
-----------------
Mirrors the BD-state code (BD_state.py): the two files share the unit convention 
and the configuration block (Part A), and the physics kernels (Part B) use 
identical formulas, so the BD benchmark that the this pipeline integrates 
against is exactly the one documented in the BD-state code. 
Physics kernels (Part B) and the numerical pipeline (Part C) are plain 
functions. Each verification test (Part D) is a function that only COMPUTES 
and returns a TestResult (a bare container: number, name, preformatted body); 
it never prints. main() calls the tests in dependency order, stores the 
results, and prints the report in logical order 1..8. To add a test: write a 
function returning a TestResult and add one line in main(). All modes are 
integrated through the single helper integrate_mode(), and every freeze-out 
location comes from mode_grid(), so the tests always probe exactly what the 
production pipeline does. 

Changing the potential
----------------------
All V(phi) dependence lives in Part A (the A_s normalisation of the scale
parameter m_phi, and the N_star / N_end conventions) and in the Part-B
kernels epsilon_of_N, eta_V_of_N (plus xi2_V_of_N in the BD code). Every
derived constant is computed from these kernels, and the ODE pipeline,
verification tests, fitting and scan are potential-independent. After a
potential swap, re-run Test 1: the atol rationale scales with H_star.

NUMERICAL ACCURACY NOTE: the mode amplitude is tiny in Planck units, 
|v| ~ k^{-1/2} ~ 1e-11, so a insufficient absolute tolerance would 
silently take over the error control (rtol*|v| ~ 1e-24) and limit the 
effective relative accuracy. We therefore use atol=1e-22. Convergence 
is verified explicitly in Test 1. 

Author: Huangruixi Lyu
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1, psi
from scipy.integrate import solve_ivp
from pathlib import Path
import sys, contextlib


# PART A. CONFIGURATION
# Unit convention: identical to the BD-state code. All internal (background) 
# quantities are in Planck units (M_pl = 1). Wavenumbers take Mpc^-1 unit only 
# through the dimensionless ratio kappa = k/k_pivot. The pipeline is scale-free, 
# so no explicit Mpc <-> M_pl conversion is needed. m_phi sets only the overall 
# amplitude and is fixed by A_s. 
N_star = 60
k_pivot = 0.05
A_s_Planck = 2.1e-9
N_end = 120
m_phi = np.pi * np.sqrt(6.0 * A_s_Planck) / (N_end - N_star)  # from A_s = m^2 (N_end - N_*)^2 / (6 pi^2)

# Integration tolerances: atol must be << |v| ~ 1e-11, see header note.
RTOL = 1e-13
ATOL = 1e-22

# G vacuum test parameters
beta_0 = 0.10
k_c = 0.50

# Planck constraints
ns_planck = 0.9649
sigma_ns = 0.0042

# Scan grids
beta_0s = [0.01, 0.03, 0.05, 0.10, 0.30]
k_cs = [0.05, 0.10, 0.30, 0.50, 1.00]

# Representative wavenumbers for Tests 6-8
EXPERIMENT_K = [1e-4, 1e-2, 0.05, 0.25, 1.0, 10.0]

# Output directory: the result/ folder next to the codes/ folder
RESULT_DIR = Path(__file__).resolve().parent.parent / "result" / "G_state"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
FIG_PZETA = RESULT_DIR / 'G_Pzeta.png'
FIG_SCAN = RESULT_DIR / 'G_parameter_scan.png'
FIG_TOL = RESULT_DIR / 'G_tolerance.png'

# PART B. PHYSICS KERNELS

# B1. Background
def epsilon_of_N(N):
    """Slow-roll parameter epsilon(N) = 1/(2(N_end - N)) for m^2 phi^2."""
    return 1.0 / (2.0 * (N_end - N))

def z_of_N(N):
    """z(N) = a(N) * sqrt(2*epsilon(N)), M_pl = 1."""
    return np.exp(N) * np.sqrt(2.0 * epsilon_of_N(N))

def eta_V_of_N(N):
    """eta_V(N). For V ~ phi^2, eta_V = epsilon_V."""
    return epsilon_of_N(N)

def nu_of_N(N):
    """Hankel index nu(N) = 3/2 + 3 eps_V - eta_V (first-order slow roll)."""
    return 1.5 + 3.0 * epsilon_of_N(N) - eta_V_of_N(N)

epsilon_star = epsilon_of_N(N_star)
nu_BD = nu_of_N(N_star)
phi_star = 2.0 * np.sqrt(N_end - N_star)
H_star = np.sqrt(0.5 * m_phi**2 * phi_star**2 / 3.0)

# B2. Bogoliubov parameters
def beta_param(k_Mpc, b0, kc):
    """Bogoliubov beta_k with Gaussian cutoff (negative for enhancement)."""
    return -b0 * np.exp(-k_Mpc**2 / (2.0 * kc**2))

def alpha_param(k_Mpc, b0, kc):
    """Bogoliubov alpha_k from the normalisation |alpha|^2 - |beta|^2 = 1."""
    return np.sqrt(1.0 + beta_param(k_Mpc, b0, kc)**2)

# B3. BD vacuum mode function (Hankel, constant nu)
def BD_mode(k, eta, nu):
    """
    BD vacuum mode function v_k^BD(eta) using Hankel H_nu^(1). 
    v_k(eta) = (sqrt(pi)/2) * e^{i*pi*(2*nu+1)/4} * sqrt(-eta) * H_nu^(1)(-k*eta)
    """
    x = max(-k * eta, 1e-300)
    C = np.sqrt(np.pi) / 2.0 * np.exp(1j * np.pi * (2.0 * nu + 1.0) / 4.0)
    return C * np.sqrt(-eta) * hankel1(nu, x)

def BD_derivative(k, eta, nu):
    """
    dv_k^BD/deta using the Hankel recurrence: 
    v' = C/sqrt(-eta) * [(nu - 1/2) * H_nu(x) - x * H_{nu-1}(x)]
    """
    x = max(-k * eta, 1e-300)
    C = np.sqrt(np.pi) / 2.0 * np.exp(1j * np.pi * (2.0 * nu + 1.0) / 4.0)
    H_nu = hankel1(nu, x)
    H_nu_m1 = hankel1(nu - 1.0, x)
    return C / np.sqrt(-eta) * ((nu - 0.5) * H_nu - x * H_nu_m1)

# B4. MS equation in x = -k*eta (two pump-field versions)
def zdd_over_z_exact(x, k):
    """
    z''/z in the quasi-de Sitter background with running slow-roll parameters: 
    z''/z = (k/x)^2 (nu(N)^2 - 1/4) = (k/x)^2 (2 + 9 eps_V - 3 eta_V) 
    truncated at first order,  N = ln(k/(H_star * x)). 
    For m^2 phi^2 (eta_V = eps_V) this reduces to 2 (k/x)^2 (1 + 3 eps). 
    """
    if x <= 0:
        return 0.0
    N = np.log(k / (H_star * x))
    if N <= 0:
        return 0.0
    return (k / x)**2 * (2.0 + 9.0 * epsilon_of_N(N) - 3.0 * eta_V_of_N(N))

def MS_ode(x, y, k):
    """
    MS equation v_xx + (1 - z''/z/k^2) v = 0 with the drifting 
    pump field z''/z.
    """
    if x <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    c = 1.0 - zdd_over_z_exact(x, k) / k**2
    return [y[2], y[3], -c * y[0], -c * y[1]]

def MS_ode_const_nu(x, y, k):
    """
    MS equation with nu frozen at nu_BD: z''/z = (k/x)^2 (nu_BD^2 - 1/4). 
    This is exactly the Hankel equation, so the same integrator, initial 
    data and readout must reproduce BD_mode() to integrator precision 
    (verified by the const-nu column of Test 6). 
    """
    if x <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    c = 1.0 - (nu_BD**2 - 0.25) / x**2
    return [y[2], y[3], -c * y[0], -c * y[1]]

# PART C. NUMERICAL PIPELINE
def mode_grid(k_Mpc, x_init=100.0):
    """
    Single source of truth for a mode's integration grid. 

    Horizon crossing: N_hc = N* + ln(kappa); freeze-out 3 e-folds later, 
    N_fz = min(N_hc + 3, 75) (the cap, 75, is never active for k <= 10 Mpc^-1, 
    max N_fz = 68.3 < 75). x_fz = k_phys e^{-N_fz}/H_star = e^-3 whenever 
    the cap is inactive -- hence identical for every mode (used by Test 7c). 
    """
    kappa = k_Mpc / k_pivot
    k_phys = kappa * np.exp(N_star) * H_star
    N_hc = N_star + np.log(kappa)
    N_fz = min(N_hc + 3.0, 75.0)
    x_fz = k_phys * np.exp(-N_fz) / H_star
    return dict(kappa=kappa, k_phys=k_phys, N_hc=N_hc, N_fz=N_fz,
                x_init=x_init, x_fz=x_fz)

def integrate_mode(k_Mpc, b0=0.0, kc=k_c, rhs=MS_ode,
                   rtol=RTOL, atol=ATOL, x_init=100.0, dense_output=False):
    """
    Integrate one mode from x_init (deep inside the horizon) to freeze-out. 

    Initial data: the BD Hankel mode, Bogoliubov-transformed with 
    (alpha_k, beta_k), reducing to the BD mode itself with b0 = 0. 
    rhs selects the pump field: MS_ode (drifting nu) or MS_ode_const_nu 
    (Hankel equation). Returns a dict with the grid, the full solution 
    object and v at freeze-out. 
    """
    g = mode_grid(k_Mpc, x_init)
    k_phys = g['k_phys']
    eta_0 = -g['x_init'] / k_phys

    v0 = BD_mode(k_phys, eta_0, nu_BD)
    vx0 = -BD_derivative(k_phys, eta_0, nu_BD) / k_phys  # v_x = -v_eta / k
    a_k = alpha_param(k_Mpc, b0, kc)
    b_k = beta_param(k_Mpc, b0, kc)
    vA = a_k * v0 + b_k * np.conj(v0)
    vxA = a_k * vx0 + b_k * np.conj(vx0)
    y0 = [vA.real, vA.imag, vxA.real, vxA.imag]

    sol = solve_ivp(lambda x, y: rhs(x, y, k_phys),
                    t_span=[g['x_init'], g['x_fz']], y0=y0,
                    method='DOP853', rtol=rtol, atol=atol,
                    dense_output=dense_output)
    v_fz = sol.y[0][-1] + 1j * sol.y[1][-1]
    return dict(grid=g, sol=sol, v_fz=v_fz)

def compute_Pzeta_G(k_Mpc, b0, kc, x_init=100.0, rtol=RTOL, atol=ATOL):
    """
    P_zeta(k) for the G vacuum at freeze-out (ODE), with the Hankel 
    (constant-nu) BD spectrum at the same readout point as reference. 
    """
    r = integrate_mode(k_Mpc, b0, kc, rtol=rtol, atol=atol, x_init=x_init)
    g = r['grid']
    k_phys, z_fz = g['k_phys'], z_of_N(g['N_fz'])
    P_G = (k_phys**3 / (2.0 * np.pi**2)) * (np.abs(r['v_fz']) / z_fz)**2
    vBD_fz = BD_mode(k_phys, -g['x_fz'] / k_phys, nu_BD)
    P_BD = (k_phys**3 / (2.0 * np.pi**2)) * (np.abs(vBD_fz) / z_fz)**2
    return P_G, P_BD

def compute_Pzeta_analytical(k_arr, b0, kc):
    """
    Analytical estimate (constant-nu) of P_zeta for the G vacuum: 
    P_G(k) = (alpha_k + |beta_k|)^2 * A_s * (k/k_pivot)^(n_s_BD - 1). 
    """
    P = np.zeros_like(k_arr)
    for i, k in enumerate(k_arr):
        enh = (alpha_param(k, b0, kc) + abs(beta_param(k, b0, kc)))**2
        n_s_BD_star = 1.0 - 6.0 * epsilon_star + 2.0 * eta_V_of_N(N_star)
        P[i] = enh * A_s_Planck * (k / k_pivot)**(n_s_BD_star - 1.0)
    return P

def fit_ns_alphas(k_arr, P_arr, kp=0.05, fit_width=5.0):
    """
    Fit n_s and alpha_s from P_zeta(k): quadratic fit of ln P vs ln(k/kp) 
    around the pivot. n_s - 1 = b, alpha_s = 2c. 
    """
    mask = (k_arr > kp / fit_width) & (k_arr < kp * fit_width)
    if np.sum(mask) < 5:
        return np.nan, np.nan
    ln_k = np.log(k_arr[mask] / kp)
    ln_P = np.log(P_arr[mask])
    coeffs = np.polyfit(ln_k, ln_P, 2)
    return 1.0 + coeffs[1], 2.0 * coeffs[0]

# PART D. VERIFICATION TESTS
# Each function only computes and returns a TestResult; it never prints.
@dataclass
class TestResult:
    """Bare container for one verification test's preformatted output."""
    number: int
    name: str
    body: str

def mode_dev_at_freezeout(k_Mpc, rhs):
    """
    dev = |v_ODE(x_fz)|^2 / |v_Hankel(x_fz)|^2 - 1 for a BD-initialised mode 
    integrated with pump field `rhs` (z cancels in the ratio). Real ODE 
    solution throughout. BD_mode() (scipy hankel1) is the exact solution of 
    the constant-nu equation, itself validated against the ODE to the 
    integrator round-off error bound by the const-nu column of Test 6. 
    """
    r = integrate_mode(k_Mpc, 0.0, k_c, rhs=rhs)
    g = r['grid']
    vH = BD_mode(g['k_phys'], -g['x_fz'] / g['k_phys'], nu_BD)
    return abs(r['v_fz'])**2 / abs(vH)**2 - 1.0

# Integrator accuracy: "error"

def test_01_tolerance_convergence(ctx):
    """
    Test 1: numerical error of the integrator under atol tightening. 
    Because |v| ~ 1e-11 in Planck units, an atol larger than rtol*|v| 
    silently degrades the effective relative accuracy. This test shows the 
    pivot spectrum ratio is stable once the atol branch contributes below 
    the rtol-limited round-off error bound (about atol ~1e-21. 
    We work at 1e-22). The plot is built in this test, showing both the 
    convergence of the error (to 1.001244 not to 1) and that it's rational 
    to choose atol = 1e-22 instead of rtol*|v| ~ 1e-24. 
    """
    P_a0, P_b0 = compute_Pzeta_G(k_pivot, 0.0, 0.5, atol=1e-22)
    R0 = P_a0 / P_b0
    lines = [f"benchmark: atol = 1e-22, R = P_G/P_BD = {R0:.7f}",
                 "error = R(atol)/R(1e-22) - 1   [numerical error, percent]"]
    ps, errs = [], []
    for p in range(15, 22):
        at = 10.0**(-p)
        P_a, P_b = compute_Pzeta_G(k_pivot, 0.0, 0.5, atol=at)
        R = P_a / P_b
        ps.append(p)
        errs.append(100 * (R - R0) / R0)
        lines.append(f"  atol = {at:8.0e}: R = {R:.7f}, "
                             f"error = {100 * (R - R0) / R0:.7f}%")
    ctx['t1_p'] = ps            # -log10(atol)
    ctx['t1_err'] = errs        # signed error in percent

    return TestResult(1, "Tolerance convergence (pivot, beta_0 = 0)",
                      "\n".join(lines))

def test_02_wronskian():
    """
    Test 2: Wronskian conservation and normalisation along the trajectory. 
    W = Im(v* v_x) = 1/(2k) is an exact invariant of the MS equation. 
    W itself is tiny (~1e-22 in these units), so both criteria are relative: 
    the normalisation error anchors W to its theory value, the CV checks 
    conservation along the trajectory. 
    Sign convention: x = -k*eta, so v ~ e^{+ix}/sqrt(2k), Im(v* v_x) > 0. 
    """
    r = integrate_mode(k_pivot, beta_0, k_c, dense_output=True)
    g = r['grid']
    xs = np.linspace(g['x_init'], g['x_fz'], 10)
    v = r['sol'].sol(xs)
    W = np.array([np.imag(np.conj(v[0][i] + 1j * v[1][i])
                          * (v[2][i] + 1j * v[3][i])) for i in range(len(xs))])
    W_theory = 1.0 / (2.0 * g['k_phys'])
    norm_err = abs(np.mean(W) / W_theory - 1.0)
    CV = np.std(W) / abs(np.mean(W))
    lines = [f"  Wronskian Im(v*, v_x): mean = {np.mean(W):.6e}, "
             f"theory = {W_theory:.6e}",
             f"  normalisation error = |mean/theory - 1| = {norm_err:.2e}",
             f"  conservation error  = std/mean (CV along trajectory) = {CV:.2e}",
             f"  Conservation: {'PASS' if CV < 1e-9 else 'FAIL'}"]
    return TestResult(2, "Wronskian conservation and normalisation",
                      "\n".join(lines))

# Analytical-vs-numerical deviations: "dev"

def test_03_pivot_single_mode():
    """
    Test 3: dev at the pivot between the ODE enhancement and the leading-order 
    formula (alpha+|beta|)^2, for the three test beta_0 values. 
    """
    lines = ["dev = (P_G/P_BD)_ODE / (alpha_k+|beta_k|)^2 - 1   "
             "[enhancement factor at the pivot, percent]"]
    for b0 in (0.03, 0.10, 0.30):
        P_a, P_b = compute_Pzeta_G(k_pivot, b0, k_c)
        R = P_a / P_b
        enh = (alpha_param(k_pivot, b0, k_c)
               + abs(beta_param(k_pivot, b0, k_c)))**2
        lines.append(f"  beta_0 = {b0:.2f}: P_G/P_BD = {R:.6f}, "
                     f"(a+|b|)^2 = {enh:.6f}, |dev| = {abs(R/enh - 1)*100:.3f}%")
    return TestResult(3, "Single mode at the pivot vs (alpha+|beta|)^2",
                      "\n".join(lines))

def test_04_full_range_deviation(ctx):
    """
    Test 4: dev over the full computed k range, reusing the Fig. 1 spectra 
    (no extra integrations). Quantifies the 'percent level' claim and its 
    k-dependence (sign change at the pivot). 
    """
    k_array = ctx['k_array']
    fit_mask = (k_array >= k_pivot / 5.0) & (k_array <= k_pivot * 5.0)
    lines = ["dev = (P_G/P_BD)_ODE / (alpha_k+|beta_k|)^2 - 1,   "
             "[enhancement factor, percent]"]
    for tag, b0 in [('003', 0.03), ('010', 0.10), ('030', 0.30)]:
        dev = np.abs(ctx[f'P_G_{tag}'] / ctx['P_BD_num'] / ctx[f'enh_{tag}'] - 1.0)
        i = int(np.argmax(dev))
        lines.append(f"  beta_0 = {b0:.2f}: max dev = {dev.max()*100:.3f}% "
                     f"(at k = {k_array[i]:.2e}), "
                     f"max dev in fit window [{k_pivot/5:.3f}, {5*k_pivot:.2f}] = "
                     f"{dev[fit_mask].max()*100:.3f}%")
    return TestResult(4, "ODE vs analytical over k in [1e-4, 10] Mpc^-1",
                      "\n".join(lines))

def test_05_beta0_limit():
    """
    Test 5: G -> BD limit. As beta_0 -> 0 the spectrum ratio must approach 1. 
    The residual at exactly beta_0 = 0 is the pure nu-drift dev 
    (isolated by Tests 6-8). 
    """
    lines = ["dev = P_G/P_BD - 1   [spectrum ratio at the pivot; expect 0]",
             "  (at beta_0 = 0 the dev is the pure nu-drift effect, see Tests 6-8)"]
    for b0 in [1e-2, 1e-4, 1e-6, 0.0]:
        P_a, P_b = compute_Pzeta_G(k_pivot, b0, 0.5)
        R = P_a / P_b
        lines.append(f"  beta_0 = {b0:8.1e}: P_G/P_BD = {R:.6f}, "
                     f"dev = {(R - 1)*100:+.4f}%")
    return TestResult(5, "beta_0 -> 0 limit (G -> BD)", "\n".join(lines))

def test_06_const_nu_experiment(ctx):
    """
    Test 6: origin of the ODE-vs-analytical dev -- constant-nu experiment. 
    The Fig. 1 dots-vs-line dev equals P_BD^ODE/P_BD^Hankel - 1 at 
    beta_0 = 0, because the enhancement is common to numerator and denominator. 
    The two runs below differ ONLY in the pump field: 
        drifting nu : z''/z = 2(k/x)^2 (1 + 3 eps(N(x)))   [production RHS] 
        constant nu : z''/z = (k/x)^2 (nu_BD^2 - 1/4)      [exactly Hankel eq.] 
    With constant nu, the ODE *is* the Hankel equation, so the same 
    integrator, initial data and readout must reproduce the Hankel value to 
    integrator precision. The drifting-nu column is therefore due to the 
    drift of nu(N) = 3/2 + 3 eps_V(N) - eta_V(N) alone. BD initial data are 
    used so the dev measures the pure amplitude response |v|^2, no interference 
    term. 
    """
    lines = ["dev_|v|^2 = |v_ODE(x_fz)|^2 / |v_Hankel(x_fz)|^2 - 1   "
             "[mode amplitude, BD initial data]",
             f"  {'k [Mpc^-1]':>10} | {'drifting nu':>12} | {'const nu':>10}"]
    experiment_dev, max_const = [], 0.0
    for k in EXPERIMENT_K:
        d_drift = mode_dev_at_freezeout(k, MS_ode)
        d_const = mode_dev_at_freezeout(k, MS_ode_const_nu)
        experiment_dev.append(d_drift)
        max_const = max(max_const, abs(d_const))
        lines.append(f"  {k:>10.4g} | {d_drift*100:>+11.4f}% | {d_const:>+10.4e}")
    lines.append(f"  Experiment: {'PASS' if max_const < 1e-8 else 'FAIL'} "
                 f"(max |const-nu dev| {max_const:.1e} = integrator round-off error bound)")
    ctx['experiment_k'] = list(EXPERIMENT_K)
    ctx['experiment_dev'] = experiment_dev
    return TestResult(6, "Constant-nu experiment (origin of the deviation)",
                      "\n".join(lines))

def test_07_freezeout_exclusion(ctx):
    """
    Test 7: freeze-out truncation is NOT the origin of the dev. 

    All numbers come from exact scipy.hankel1 evaluations at x_fz = e^-3 
    and x_ref = e^-30 (where the subleading branch of the small-x expansion 
    is suppressed to x^{2 nu} ~ 1e-40); the series 
    H = -(i/pi) Gamma(nu) (2/x)^nu [1 + (x/2)^2/(nu-1) + ...] 
    appears only as a printed cross-check, never as a computed value. 

    (a) deviation from the leading small-x asymptotic form: the asymptotic 
        coefficient is MEASURED at x_ref and scaled back to x_fz with the 
        exact x^{1/2-nu} scaling of the leading branch (the amplitude itself 
        grows as x^{1/2-nu}, so the comparison must be transported to the 
        same x -- a direct ratio |v(x_fz)|^2/|v(x_ref)|^2 would just measure 
        the 27-e-fold growth, not the residual); the residual at x_fz is 
        common to the G and BD spectra (same readout point), so it cancels 
        in their ratio; 
    (b) the phase enters the enhancement as enh = a^2 + b^2 + 2ab cos(2phi): 
        (b1) freeze-out drift dphi = arg v(x_fz) - arg v(x_ref): the actual 
             truncation error; its effect on enh is suppressed because 2phi 
             sits near the cos extremum; 
        (b2) asymptotic offset: with the standard BD phase, 
             v*/v -> -exp(-i pi (nu - 3/2)), not exactly -1, so analytical
             formula carries an O(eps*^2) correction even at perfect 
             freeze-out -- a property of the analytical formula, not of the 
             readout, accounting for the small beta_0 dependence in Test 4; 
    (c) x_fz comes from mode_grid(), the same helper the pipeline uses: 
        it is e^-3 for every mode (cap inactive for k <= 10), so any 
        freeze-out residual would be k-independent -- incompatible with the 
        sign-changing pattern seen in Tests 4 and 6. 
    """
    nu = nu_BD
    C = np.sqrt(np.pi) / 2.0 * np.exp(1j * np.pi * (2.0 * nu + 1.0) / 4.0)
    x_fz, x_ref = np.exp(-3.0), np.exp(-30.0)
    v_fz = C * np.sqrt(x_fz) * hankel1(nu, x_fz)
    v_ref = C * np.sqrt(x_ref) * hankel1(nu, x_ref)

    # (a) asymptotic coefficient measured at x_ref, scaled back to x_fz
    p_asym = 0.5 - nu        # exact x-scaling of the leading asymptotic branch
    A_asym = abs(v_ref) / x_ref**p_asym                 # numerical measurement
    amp_resid = abs(v_fz)**2 / (A_asym * x_fz**p_asym)**2 - 1.0
    amp_series = 2.0 * (x_fz / 2.0)**2 / (nu - 1.0)     # cross-check only

    # (b) phases from the exact evaluations
    a_t = alpha_param(k_pivot, beta_0, k_c)
    b_t = beta_param(k_pivot, beta_0, k_c)
    phi_fz, phi_ref = np.angle(v_fz), np.angle(v_ref)
    dphi = phi_fz - phi_ref                             # freeze-out drift only

    def enh_from_phase(c2phi):
        """|a + b e^{-2i phi}|^2 with |v|^2 factored out; c2phi = cos(2 phi)."""
        return a_t**2 + b_t**2 + 2.0 * a_t * b_t * c2phi

    enh_fz = enh_from_phase(np.cos(2.0 * phi_fz))       # as readout at x_fz
    enh_x0 = enh_from_phase(np.cos(2.0 * phi_ref))      # perfect freeze-out
    enh_formula = (a_t + abs(b_t))**2                   # analytical formula assumes v*/v = -1 exactly

    # (c) freeze-out locations
    x_fz_all = [mode_grid(k)['x_fz'] for k in ctx['experiment_k']]

    lines = [
        f"  (a) dev_|v|^2 = |v(x_fz)|^2/|v_asym(x_fz)|^2 - 1 = {amp_resid:+.4%}",
        f"      [asymptotic coefficient measured at x_ref = e^-30, scaled back "
        f"with the exact x^(1/2-nu) asymptotics; series cross-check "
        f"2(x_fz/2)^2/(nu-1) = {amp_series:+.4%}]",
        "      common to G and BD (same readout point) -> cancels in the ratio",
        f"  (b1) dphi = arg v(x_fz) - arg v(x_ref) = {dphi:.2e} rad",
        f"      dev_enh = enh(phi_fz)/enh(phi_ref) - 1 = {enh_fz/enh_x0 - 1.0:+.2e},"
        f"  enh(phi) = a^2+b^2+2ab cos(2phi)",
        f"  (b2) dev_enh = enh(phi_ref)/(a+|b|)^2 - 1 = {enh_x0/enh_formula - 1.0:+.2e}",
        "      (O(eps*^2) property of the analytical formula: v*/v -> -exp(-2 i pi eps*),",
        "       independent of x_fz; NOT a truncation error)",
        f"  (c) x_fz from mode_grid() for the Test-6 modes: "
        f"{['%.4f' % x for x in x_fz_all]} -- all e^-3, hence k-independent",
    ]
    return TestResult(7, "Freeze-out truncation exclusion", "\n".join(lines))

def test_08_analytic_estimate(ctx):
    """
    Test 8: closed-form estimate vs measured dev. 
    Superhorizon Hankel amplitude |v|^2 -> Gamma(nu)^2 2^(2nu-2)/pi x^(1-2nu) 
    gives the sensitivity S = d ln|v|^2/d nu = 2 psi(nu) + 2 ln2 - 2 ln x_fz 
    (~7.5, dominated by -2 ln x_fz = 6: the leading small-x amplitude 
    |v|^2 ~ x^{1-2nu} accumulates three e-folds of growth between horizon 
    crossing and freeze-out, so an exponent mismatch is amplified 
    exponentially). A mode crossing at 
    N_hc = N* + ln(k/k*) gives delta_nu = nu(N_hc) - nu*, so 
        dev(k) ~= S * 2 [eps(N_hc) - eps*], 
    first order in epsilon, sign = sign(ln(k/k*)) -> negative at low k, 
    positive at high k. 
    The estimate column is the closed-form prediction (background quantities 
    only); the measured column is the Test-6 ODE result. 
    """
    x_fz_ref = np.exp(-3.0)
    S = 2.0 * psi(nu_BD) + 2.0 * np.log(2.0) - 2.0 * np.log(x_fz_ref)
    lines = [f"  S = d ln|v|^2/d nu = {S:.2f} "
             f"(dominated by -2 ln x_fz = {-2.0*np.log(x_fz_ref):.1f})",
             "  estimate = S [nu(N_hc) - nu*],  N_hc = N* + ln(k/k*)",
             f"  {'k [Mpc^-1]':>10} | {'estimate':>9} | {'measured (Test 6)':>17}"]
    for k, d in zip(ctx['experiment_k'], ctx['experiment_dev']):
        est = S * (nu_of_N(N_star + np.log(k / k_pivot)) - nu_BD)
        # est = S * 2.0 * (epsilon_of_N(N_star + np.log(k / k_pivot))
        #                  - epsilon_star)
        lines.append(f"  {k:>10.4g} | {est*100:>+8.4f}% | {d*100:>+16.4f}%")
    return TestResult(8, "Closed-form estimate vs measured deviation",
                      "\n".join(lines))

def print_test_report(results):
    """Print all test results in logical (numbered) order."""
    print("\n--- VERIFICATION TESTS ---")
    print("  naming: 'error' = numerical inaccuracy of the integrator itself;")
    print("          'dev'   = (numerical - analytical)/analytical, a physical")
    print("          difference between approximation schemes, not a mistake")
    for n in sorted(results):
        r = results[n]
        print(f"\n[Test {r.number}] {r.name}")
        print(r.body)

# PART E. SPECTRA + FIGURE 1
def compute_spectra(ctx):
    """ODE spectra for the three test beta_0 values + analytical curves."""
    k_array = np.logspace(-4, 1, 80)
    ctx['k_array'] = k_array
    for tag, b0 in [('003', 0.03), ('010', 0.10), ('030', 0.30)]:
        res = np.array([compute_Pzeta_G(k, b0, k_c) for k in k_array])
        ctx[f'P_G_{tag}'] = res[:, 0]
        # Hankel (constant-nu) BD spectrum at x_fz, same for all beta_0
        ctx['P_BD_num'] = res[:, 1]
        ctx[f'enh_{tag}'] = (alpha_param(k_array, b0, k_c)
                             + np.abs(beta_param(k_array, b0, k_c)))**2
    n_s_BD_star = 1.0 - 6.0 * epsilon_star + 2.0 * eta_V_of_N(N_star)
    ctx['P_BD_ana'] = A_s_Planck * (k_array / k_pivot)**(n_s_BD_star - 1.0)
    for tag in ['003', '010', '030']:
        ctx[f'P_ana_{tag}'] = ctx[f'enh_{tag}'] * ctx['P_BD_ana']

def make_figure_spectrum(ctx):
    """Fig. 1: P_zeta(k) and enhancement factor for the test parameters."""
    k_array = ctx['k_array']
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P_zeta(k) -- analytical spectra (not the ODE absolute spectra)
    ax = axes1[0]
    ax.loglog(k_array, ctx['P_BD_ana'], 'k-', lw=2.5, label='BD vacuum')
    ax.loglog(k_array, ctx['P_ana_003'], 'b-', lw=1.5, label=r'$\beta_0=0.03$')
    ax.loglog(k_array, ctx['P_ana_010'], 'g-', lw=1.5, label=r'$\beta_0=0.10$')
    ax.loglog(k_array, ctx['P_ana_030'], 'r-', lw=1.5, label=r'$\beta_0=0.30$')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel(r'$P_\zeta$', fontsize=14)
    ax.set_title(r'$P_\zeta$, $k_c = 0.5$ Mpc$^{-1}$', fontsize=15)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    # Right: enhancement factor -- ODE same-pipeline ratio (dots) vs analytical (lines)
    ax = axes1[1]
    ax.semilogx(k_array, ctx['P_G_003'] / ctx['P_BD_num'], 'bo', ms=3, alpha=0.6)
    ax.semilogx(k_array, ctx['P_G_010'] / ctx['P_BD_num'], 'go', ms=3, alpha=0.6)
    ax.semilogx(k_array, ctx['P_G_030'] / ctx['P_BD_num'], 'ro', ms=3, alpha=0.6)
    ax.semilogx(k_array, ctx['enh_003'], 'b-', lw=2, label=r'$\beta_0=0.03$')
    ax.semilogx(k_array, ctx['enh_010'], 'g-', lw=2, label=r'$\beta_0=0.10$')
    ax.semilogx(k_array, ctx['enh_030'], 'r-', lw=2, label=r'$\beta_0=0.30$')
    ax.axhline(1.0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel(r'$P_\zeta^{\rm G} / P_\zeta^{\rm BD}$', fontsize=14)
    ax.set_title(r'Enhancement Factor $(\alpha_k+|\beta_k|)^2$, $k_c = 0.5$ Mpc$^{-1}$',
                 fontsize=15)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_PZETA, dpi=200)
    print("\n[Saved: Pzeta.png]")

# PART F. PARAMETER SCAN + FIGURE 2
def run_parameter_scan():
    """Full 5x5 scan over (beta_0, k_c) with n_s fitting and Planck chi^2."""
    print("\n" + "=" * 60)
    print("FULL PARAMETER SCAN (5x5 grid)")
    print("=" * 60)

    k_scan = np.logspace(-3, 0.7, 200)
    results = {}
    n_s_BD_star = 1.0 - 6.0 * epsilon_star + 2.0 * eta_V_of_N(N_star)
    ns_bd = n_s_BD_star

    for b0 in beta_0s:
        for kc in k_cs:
            P_G = compute_Pzeta_analytical(k_scan, b0, kc)
            ns_fit, als = fit_ns_alphas(k_scan, P_G, kp=k_pivot, fit_width=5.0)
            chi2 = (ns_fit - ns_planck)**2 / sigma_ns**2 if not np.isnan(ns_fit) else np.inf
            results[(b0, kc)] = {
                'ns': ns_fit, 'alpha_s': als, 'chi2': chi2,
                'delta_ns': ns_fit - ns_bd if not np.isnan(ns_fit) else np.nan,
            }
            delta_ns_naive = -2.0 * b0 * (k_pivot ** 2) / (kc ** 2)
            delta_ns_fit = ns_fit - ns_bd
            marker = " ***" if chi2 < 4.0 else ""
            print(f"  b0={b0:5.2f}, kc={kc:5.2f}: "
                  f"ns_fit={ns_fit:.4f} (BD={ns_bd:.4f}),   "
                  f"d_fit={delta_ns_fit:+.4f}, d_naive={delta_ns_naive:+.4f},   "
                  f"d_fit/d_naive={delta_ns_fit / delta_ns_naive:+.4f}, "
                  f"as={als:.6f}, chi2={chi2:.2f}{marker}")

    print("\n*** = within 2sigma of Planck (chi2 < 4)")
    return results

def make_figure_scan(results):
    """Fig. 2: parameter scan results (n_s, Delta n_s, chi^2)."""
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

    # n_s vs beta_0
    ax = axes2[0, 0]
    for kc in k_cs:
        ns_vals = [results[(b0, kc)]['ns'] for b0 in beta_0s]
        ax.plot(beta_0s, ns_vals, 'o-', lw=2, markersize=6, label=f'$k_c={kc}$')
    ax.axhline(ns_planck, color='black', ls='--', alpha=0.7,
               label=f'Planck $n_s={ns_planck:.4f}$')
    ax.axhspan(ns_planck - 2.0 * sigma_ns, ns_planck + 2.0 * sigma_ns,
               alpha=0.15, color='green', label='Planck 95% CL')
    ax.set_xlabel(r'$\beta_0$', fontsize=14)
    ax.set_ylabel(r'$n_s$', fontsize=14)
    ax.set_title(r'$n_s$ vs $\beta_0$', fontsize=15)
    ax.legend(loc='lower left', fontsize=14)
    ax.grid(True, ls='--', alpha=0.5)

    # n_s vs k_c
    ax = axes2[0, 1]
    for b0 in beta_0s:
        ns_vals = [results[(b0, kc)]['ns'] for kc in k_cs]
        ax.plot(k_cs, ns_vals, 'o-', lw=2, markersize=6, label=f'$\\beta_0={b0}$')
    ax.axhline(ns_planck, color='black', ls='--', alpha=0.7)
    ax.axhspan(ns_planck - 2.0 * sigma_ns, ns_planck + 2.0 * sigma_ns,
               alpha=0.15, color='green')
    ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel(r'$n_s$', fontsize=14)
    ax.set_title(r'$n_s$ vs $k_c$', fontsize=15)
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True, ls='--', alpha=0.5)

    # delta_n_s heatmap
    ax = axes2[1, 0]
    delta_ns_mat = np.array([[results[(b0, kc)]['delta_ns'] for kc in k_cs]
                             for b0 in beta_0s])
    im = ax.imshow(delta_ns_mat, aspect='auto', cmap='RdBu_r',
                   extent=[min(k_cs), max(k_cs), min(beta_0s), max(beta_0s)],
                   origin='lower')
    ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel(r'$\beta_0$', fontsize=14)
    ax.set_title(r'$\Delta n_s = n_s^{\rm G} - n_s^{\rm BD}$', fontsize=15)
    plt.colorbar(im, ax=ax, label=r'$\Delta n_s$')
    for i, b0 in enumerate(beta_0s):
        for j, kc in enumerate(k_cs):
            if results[(b0, kc)]['chi2'] < 4.0:
                ax.plot(kc, b0, 'w*', markersize=15, markeredgecolor='black',
                        markeredgewidth=1)

    # chi^2 exclusion map
    ax = axes2[1, 1]
    chi2_mat = np.array([[min(results[(b0, kc)]['chi2'], 50.0) for kc in k_cs]
                         for b0 in beta_0s])
    im2 = ax.imshow(chi2_mat, aspect='auto', cmap='YlOrRd',
                    extent=[min(k_cs), max(k_cs), min(beta_0s), max(beta_0s)],
                    origin='lower')
    ax.set_xlabel(r'$k_c$ [Mpc$^{-1}$]', fontsize=14)
    ax.set_ylabel(r'$\beta_0$', fontsize=14)
    ax.set_title(r'$\chi^2$ (Planck, capped at 50)', fontsize=15)
    plt.colorbar(im2, ax=ax, label=r'$\chi^2$')
    for i, b0 in enumerate(beta_0s):
        for j, kc in enumerate(k_cs):
            if chi2_mat[i, j] < 4.0:
                ax.plot(kc, b0, 'w*', markersize=15, markeredgecolor='black',
                        markeredgewidth=1)

    plt.tight_layout()
    plt.savefig(FIG_SCAN, dpi=200)
    print("\n[Saved: G_parameter_scan.png]")

def make_figure_tolerance(ctx):
    """Fig. A: Error convergence plot in Test 1(|error| vs -log10(atol))"""
    p = np.array(ctx['t1_p'])
    err = np.abs(np.array(ctx['t1_err']))
    err_plot = np.where(err > 0, err, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(p, err_plot, 'o-', lw=2, ms=7, color='C0')

    ax.axvline(22, color='gray', ls='--', alpha=0.7)
    ax.annotate('working point\natol = 1e-22', xy=(22, 1e-3), xytext=(18.8, 3e-3),
            arrowprops=dict(arrowstyle='->'), fontsize=10)
    ax.annotate('atol = 1e-15\n(~0.1%, significant in most devs)',
                xy=(15, err[0]), xytext=(16.3, 3e-2),
                arrowprops=dict(arrowstyle='->'), fontsize=10)

    ax.set_xlabel(r'$-\log_{10}(\mathrm{atol})$', fontsize=14)
    ax.set_ylabel(r'|error| = $|R(\mathrm{atol})/R(10^{-22}) - 1|$  [%]', fontsize=13)
    ax.set_title('Tolerance convergence at the pivot (Test 1)', fontsize=14)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_TOL, dpi=200)
    print(f"\n[Saved: G_tolerance.png]")

# PART G. SUMMARY + MAIN
def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nPlanck-Compatible Parameters (chi2 < 4, 95% CL):")
    compatible = []
    for (b0, kc), row in results.items():
        if row['chi2'] < 4.0:
            compatible.append((b0, kc, row['ns'], row['chi2']))
            print(f"  beta_0={b0:5.2f}, k_c={kc:5.2f}: "
                  f"n_s={row['ns']:.4f}, chi2={row['chi2']:.2f}")
    if not compatible:
        print("  (none within 95% CL)")
    print("\nAll outputs:")
    print("  G_Pzeta.png - P_zeta(k) for test parameters")
    print("  G_parameter_scan.png - 5x5 parameter scan results")
    print("=" * 60)

def main():
    print("=" * 60)
    print("G Vacuum - Complete Solution with Parameter Scan")
    print("=" * 60)
    print(f"m_phi = {m_phi:.4e} M_pl (from A_s = m^2 N_*^2/(6 pi^2))")
    print(f"H_*   = {H_star:.4e} M_pl")
    print(f"Tolerances: rtol = {RTOL:.0e}, atol = {ATOL:.0e}")

    ctx = {}
    results = {}

    # computation phase (dependency order; only progress is printed)
    print("\n>> Running verification tests ...")
    results[1] = test_01_tolerance_convergence(ctx)
    results[2] = test_02_wronskian()
    results[3] = test_03_pivot_single_mode()
    results[5] = test_05_beta0_limit()
    print(">> Computing P_zeta(k) spectra (needed by Test 4 and Fig. 1) ...")
    compute_spectra(ctx)
    results[4] = test_04_full_range_deviation(ctx)
    results[6] = test_06_const_nu_experiment(ctx)   # stores experiment results in ctx
    results[7] = test_07_freezeout_exclusion(ctx)
    results[8] = test_08_analytic_estimate(ctx)

    # report phase (logical order 1..8)
    print_test_report(results)
    make_figure_tolerance(ctx)

    # figures, scan, summary
    make_figure_spectrum(ctx)
    scan_results = run_parameter_scan()
    make_figure_scan(scan_results)
    print_summary(scan_results)

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
    log_path = RESULT_DIR / 'G_console.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file, \
            contextlib.redirect_stdout(_Tee(log_file)):
        main()
