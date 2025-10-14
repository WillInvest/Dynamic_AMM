

#%%



"""Vectorized multi-path theta simulation with per-bin counts aggregation."""

import math
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

# --- Tunables ---
STEPS   = 1_000_000
WARMUP  = 50000
DT      = 1.0 / (365.0 * 24.0 * 60 * 5)
MU      = 0.0
SIGMA   = 0.2
GAMMA   = 0.01
S0      = 1.0
X0, Y0  = 1_000_000.0, 1_000_000.0
L0      = np.sqrt(X0 * Y0)
BINS    = 500
BMIN    = -1.0
BMAX    = 1.0
EPS     = 1e-8
RUNS    = 500

rng = np.random.default_rng(int(time.time()))
diff = SIGMA * math.sqrt(DT)
drift = (MU - 0.5 * SIGMA * SIGMA) * DT

def main():
    counts = np.zeros(BINS, dtype=np.int64)
    bin_scale = BINS / (BMAX - BMIN)
    S = np.full(RUNS, 1.0)
    X = np.full(RUNS, X0)
    Y = np.full(RUNS, Y0)
    progress_bar = tqdm(total=STEPS, desc="Steps progressed")
    for t in range(STEPS):
        P = Y / X
        L = np.sqrt(X * Y)
        assert np.max(np.abs(L - L0)) < EPS, f"Constant Product breaks with new L {L}"
        z = rng.standard_normal(RUNS)
        S *= np.exp(drift + diff * z)
        ub = P / (1-GAMMA)
        lb = P * (1-GAMMA)
        hi = S > ub
        lo = S < lb
        if hi.any():
            X_hi = L[hi] / np.sqrt((1-GAMMA) * S[hi])
            Y_hi = L[hi] * np.sqrt((1-GAMMA) * S[hi])
            X[hi], Y[hi] = X_hi, Y_hi
        if lo.any():
            X_lo = L[lo] * np.sqrt((1-GAMMA)/S[lo])
            Y_lo = L[lo] * np.sqrt(S[lo]/(1-GAMMA))
            X[lo], Y[lo] = X_lo, Y_lo
        theta = np.clip(np.log(P/S) / np.log(1-GAMMA), -1, 1)
        assert np.max(np.abs(theta) - 1) < EPS, f"theta beyond [-1,1] {theta}"
        if t >= WARMUP:
            idx = ((theta - BMIN) * bin_scale).astype(np.int64)
            np.clip(idx, 0, BINS - 1, out=idx)
            np.add.at(counts, idx, 1)
            
        progress_bar.update(1)
    progress_bar.close()
    return counts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    counts = main()
    counts = counts / counts.sum()
    edges = np.linspace(BMIN, BMAX, BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (BMAX - BMIN) / BINS
    plt.bar(centers, counts, width=width, align='center', edgecolor='k')
    plt.xlim(edges[0], edges[-1])
    plt.xlabel('theta')
    plt.ylabel('count')
    plt.show()
#%%
