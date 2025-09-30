# theta_stationary_no_grid_quadrature.py
# Stationary distribution for theta with GBM-driven "clipped" dynamics (no state binning).
# Uses Gauss–Legendre quadrature to approximate integrals and fixed-point iteration
# to solve for the stationary interior pdf on (-1,1) plus endpoint atoms at -1 and +1.

import math
from dataclasses import dataclass
from typing import List
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


# ---------- Normal PDF/CDF (no SciPy needed) ----------

def phi(z: float) -> float:
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)


def Phi(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------- Kernel precomputation ----------

@dataclass
class KernelPieces:
    # Quadrature on [-1, 1]
    x: np.ndarray   # nodes (N,)
    w: np.ndarray   # weights (N,)
    # Interior kernel (with source-side weights applied)
    K: np.ndarray          # (N, N), K_{j,i} = k(x_j | x_i) * w_i
    # Interior contributions when source is an endpoint atom
    k_from_minus: np.ndarray  # (N,), k(y | -1) evaluated at y = nodes
    k_from_plus:  np.ndarray  # (N,), k(y | +1)
    # Tail probabilities (to atoms) from interior nodes, with weights
    ell_vec: np.ndarray   # (N,), ell(x_i) * w_i
    r_vec:   np.ndarray   # (N,), r(x_i) * w_i
    # Tail probabilities from endpoints
    ell_minus: float
    ell_plus: float
    r_minus: float
    r_plus: float
    # Parameters
    tau: float
    delta: float
    a: float
    sigma: float
    gamma: float
    dt: float


def build_kernel_pieces(
    sigma: float,
    gamma: float,
    dt: float,
    N: int = 128,
    include_drift: bool = False,
) -> KernelPieces:
    """
    Precompute quadrature nodes/weights and kernel objects.
      sigma: annualized GBM vol
      gamma: AMM fee parameter in (0,1)
      dt:    step in years (e.g., 1/(252*6.5*60) ~ 1 minute)
      N:     # of Gauss–Legendre nodes
      include_drift: include tiny drift 0.5*sigma^2*dt/a in theta-step mean
    """
    a = -math.log(1.0 - gamma)
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0,1)")
    if a <= 0.0:
        raise ValueError("log(1-gamma) produced nonpositive 'a'")

    tau = sigma * math.sqrt(dt) / a
    delta = (0.5 * sigma * sigma * dt / a) if include_drift else 0.0

    # Gauss–Legendre on [-1,1]
    # x, w = np.polynomial.legendre.leggauss(N)
    # N = 10  # number of points
    x = np.linspace(-1, 1, N+2)[1:-1]   # exclude -1 and 1
    w = np.full_like(x, 2.0 / N)        # uniform weights summing to 2

    def k_y_given_x(y: np.ndarray, x_scalar: float) -> np.ndarray:
        z = (y - (x_scalar - delta)) / tau
        return (1.0 / tau) * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def ell_of_x(x_scalar: float) -> float:
        z = (-1.0 - (x_scalar - delta)) / tau
        return Phi(z)

    def r_of_x(x_scalar: float) -> float:
        z = (1.0 - (x_scalar - delta)) / tau
        return 1.0 - Phi(z)

    Nn = len(x)
    K = np.empty((Nn, Nn), dtype=float)
    for i in range(Nn):
        # weight on *source* node i
        K[:, i] = k_y_given_x(x, x[i]) * w[i]

    k_from_minus = k_y_given_x(x, -1.0)
    k_from_plus  = k_y_given_x(x, +1.0)

    ell_vec = np.array([ell_of_x(xi) for xi in x]) * w
    r_vec   = np.array([r_of_x(xi) for xi in x]) * w

    ell_minus = ell_of_x(-1.0)
    ell_plus  = ell_of_x(+1.0)
    r_minus   = r_of_x(-1.0)
    r_plus    = r_of_x(+1.0)

    return KernelPieces(
        x=x, w=w, K=K,
        k_from_minus=k_from_minus, k_from_plus=k_from_plus,
        ell_vec=ell_vec, r_vec=r_vec,
        ell_minus=ell_minus, ell_plus=ell_plus, r_minus=r_minus, r_plus=r_plus,
        tau=tau, delta=delta, a=a, sigma=sigma, gamma=gamma, dt=dt
    )


# ---------- Fixed-point iteration (stationary law) ----------

@dataclass
class StationaryResult:
    x: np.ndarray
    w: np.ndarray
    f_nodes: np.ndarray
    m_left: float
    m_right: float
    tau: float
    delta: float
    a: float
    hist_L1: List[float]
    hist_total_mass_raw: List[float]


def iterate_stationary(
    kp: KernelPieces,
    max_iter: int = 4000,
    tol: float = 1e-10,
    renormalize: str = "global",  # "global" or "interior"
) -> StationaryResult:
    """
    Compute stationary (f*, m_-*, m_+*) by iterating the one-step map and renormalizing.

    renormalize:
      - "global": divide (f', m_-', m_+') by total raw mass (keeps total=1).
      - "interior": keep atoms as-computed, rescale interior so interior mass = 1 - m_- - m_+.
    """
    x, w = kp.x, kp.w
    N = len(x)

    # Start: uniform interior (pdf=1/2 on [-1,1]) and zero atoms.
    f = np.full(N, 0.5)
    mL, mR = 0.0, 0.0
    L1 = 100.0

    hist_L1, hist_M = [], []

    for _ in range(max_iter):
        # Push forward (raw)
        f_raw = kp.K @ f + mL * kp.k_from_minus + mR * kp.k_from_plus
        mL_raw = float(kp.ell_vec @ f + mL * kp.ell_minus + mR * kp.ell_plus)
        mR_raw = float(kp.r_vec   @ f + mL * kp.r_minus   + mR * kp.r_plus)

        # Raw masses (interior via quadrature)
        m_int_raw = float(np.dot(w, f_raw))
        M_raw = m_int_raw + mL_raw + mR_raw

        # Renormalize (helps kill numerical drift)
        if renormalize == "global":
            f_new  = f_raw / M_raw
            mL_new = mL_raw / M_raw
            mR_new = mR_raw / M_raw
        elif renormalize == "interior":
            target_int = max(0.0, 1.0 - (mL_raw + mR_raw))
            scaling = target_int / m_int_raw if m_int_raw > 0 else 0.0
            f_new = f_raw * scaling
            mL_new, mR_new = mL_raw, mR_raw
        else:
            raise ValueError("renormalize must be 'global' or 'interior'")

        # Clip tiny negatives due to roundoff
        f_new = np.maximum(f_new, 0.0)
        mL_new = max(mL_new, 0.0)
        mR_new = max(mR_new, 0.0)

        # L1 distance between consecutive distributions (interior + atoms)
        d_int = float(np.dot(w, np.abs(f_new - f)))
        d_atoms = abs(mL_new - mL) + abs(mR_new - mR)
        L1 = d_int + d_atoms

        hist_L1.append(L1)
        hist_M.append(M_raw)

        f, mL, mR = f_new, mL_new, mR_new

        if L1 < tol:
            break

    return StationaryResult(
        x=x, w=w, f_nodes=f, m_left=mL, m_right=mR,
        tau=kp.tau, delta=kp.delta, a=kp.a,
        hist_L1=hist_L1, hist_total_mass_raw=hist_M
    )


def analytical_incoming_fee(theta, gamma, sigma, delta_t=1/(365*24)):
    x = 1e6
    y = 1e6
    L = np.sqrt(x * y)
    p0 = (y/x) * (1-gamma)**(-theta)

    sdt = sigma * np.sqrt(delta_t)

    d1 = (np.log((1-gamma) * y / (p0 * x))) / sdt
    d2 = (np.log(y / ((1-gamma) * x * p0))) / sdt

    alpha = L * np.sqrt((1-gamma)*p0) * np.exp(- sigma**2 * delta_t / 8.0)

    incoming_fee = gamma/(1-gamma) * (
        alpha * (norm.cdf(d1) + norm.cdf(-d2))
        - p0*x*norm.cdf(d1 - sdt/2.0)
        - y*norm.cdf(-d2 - sdt/2.0)
    )
    return incoming_fee

def run(gamma, sigma):
    
    dir = f"/Users/haofu/Desktop/AMM/Dynamic_AMM/Theta/results1/gamma_{gamma:.4f}_sigma_{sigma:.2f}"
    os.makedirs(dir, exist_ok=True)
    
    dt = 12/(365*24*60*60)  # 1-minute step during trading hours

    N = 128
    include_drift = True  # usually negligible at small dt
    tol = 1e-10
    max_iter = 5000
    renorm = "global"      # or "interior"

    kp = build_kernel_pieces(sigma, gamma, dt, N=N, include_drift=include_drift)
    res = iterate_stationary(kp, max_iter=max_iter, tol=tol, renormalize=renorm)

    interior_mass = float(np.dot(res.w, res.f_nodes))
    total_mass = interior_mass + res.m_left + res.m_right

    # interior probabilities at nodes
    probs_interior = res.w * res.f_nodes

    # add atoms
    p_left = res.m_left
    p_right = res.m_right

    # sanity check: should be 1
    total_mass = probs_interior.sum() + p_left + p_right

    # normalize just in case of small numerical drift
    probs_interior /= total_mass
    p_left /= total_mass
    p_right /= total_mass
    
    plt.figure()
    plt.bar(res.x, probs_interior, width=0.02, label=r"$\theta$ $\in$ $(-1,1)$")
    plt.bar(-1, p_left, width=0.02, label=r"$\theta$ = -1")
    plt.bar(1, p_right, width=0.02, label=r"$\theta$ = 1")
    plt.xlabel(r"$\theta$")
    plt.ylabel("probability")
    plt.title(f"Stationary distribution (gamma={gamma:.4f}, sigma={sigma:.2f})")
    # make the loc auto
    plt.legend(loc="best")
    plt.savefig(f"{dir}/theta_stationary_distribution.png")

    # Plot convergence history (L1 change per iteration)
    plt.figure()
    plt.semilogy(range(1, len(res.hist_L1) + 1), res.hist_L1)
    plt.title(f"Convergence of fixed-point iteration (gamma={gamma:.4f}, sigma={sigma:.2f})")
    plt.xlabel("iteration")
    plt.ylabel("L1 change")
    plt.grid(True)
    plt.savefig(f"{dir}/theta_stationary_distribution_convergence.png")
    
    # Save CSV of stationary pdf (nodes, weights, density)
    df = pd.DataFrame({
        "theta_node": res.x,
        "weight": res.w,
        "pdf_value": res.f_nodes,
    })
    df.to_csv(f"{dir}/theta_stationary_density.csv", index=False)
    
    # Make sure fee uses same dt as kernel
    fee_dt = dt  # keep these aligned

    # Vectorized evaluation of your fee at the Gauss–Legendre nodes
    fee_at_nodes = analytical_incoming_fee(res.x, gamma=gamma, sigma=sigma, delta_t=fee_dt)

    # Interior expectation via quadrature + endpoint atoms
    E_fee_interior = float(np.dot(res.w, fee_at_nodes * res.f_nodes))
    E_fee_atoms    = res.m_left  * analytical_incoming_fee(-1.0, gamma, sigma, delta_t=fee_dt) \
                   + res.m_right * analytical_incoming_fee(+1.0, gamma, sigma, delta_t=fee_dt)

    E_fee_total = E_fee_interior + E_fee_atoms
    fee_at_mid = analytical_incoming_fee(0.0, gamma, sigma, delta_t=fee_dt)
    
    info = []
    
    info.append({
        "N": N,
        "include_drift": include_drift,
        "tol": tol,
        "max_iter": max_iter,
        "renorm": renorm,
        "gamma": gamma,
        "sigma": sigma,
        "dt": dt,
        "m_left": res.m_left,
        "m_right": res.m_right, 
        "E_fee_total": E_fee_total,
        "fee_at_mid": fee_at_mid,
    })
    
    pd.DataFrame(info).to_csv(f"{dir}/theta_info.csv", index=False)


if __name__ == "__main__":
    
    # gamma_list = np.arange(0.0101, 0.0501, 0.0001)
    # sigma_list = np.arange(0.1, 2.1, 0.1)
    sigma_list         = np.array([0.2, 0.5], dtype=np.float32)
    gamma_list         = np.array([0.0001, 0.005, 0.01], dtype=np.float32)
    total_combinations = len(gamma_list) * len(sigma_list)
    pbar = tqdm(total=total_combinations, desc="Processing parameter combinations")
    for gamma in gamma_list:
        for sigma in sigma_list:
            run(gamma, sigma)
            pbar.update(1)
    pbar.close()