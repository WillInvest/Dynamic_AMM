# theta_hist_parallel_vec.py
# Parallel streaming histograms of theta for (sigma, gamma) grid, fully vectorized over gamma.
import os, time, math, threading, queue
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# --- Tunables ---
STEPS          = 10000
WARMUP         = 200
WORKERS        = 20
DT             = 1.0 / (365.0 * 24.0 * 60 * 5)
S0             = 1.0
X0, Y0         = 1_000_000.0, 1_000_000.0
MU             = 0.0
# SIGMAS         = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], dtype=np.float32)
# GAMMAS         = np.arange(0.0001, 0.03, 0.0005, dtype=np.float32)
SIGMAS         = np.array([0.5], dtype=np.float32)
GAMMAS         = np.array([0.01], dtype=np.float32)

EPS            = 1e-8
REPORT_EVERY_STEPS = 1
SEEDS_PER_WORKER = 10

# Global theta bins
BINS    = 128
BMIN    = -1.0
BMAX    =  1.0
EDGES   = np.linspace(BMIN, BMAX, BINS + 1, dtype=np.float32)


timestamp = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"theta_hist_out/{timestamp}"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NPZ = os.path.join(OUT_DIR, f"hists_{timestamp}.npz")

# Avoid BLAS oversubscription
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")


def _worker_chunk(worker_idx: int, progress_q=None, report_every_steps: int = 0):
    """
    Simulate a worker's share of seeds.
    Returns (counts[S,G,BINS], under[S,G], over[S,G], eq_neg[S,G], eq_pos[S,G]).
    """
    S, G = SIGMAS.size, GAMMAS.size
    counts = np.zeros((S, G, BINS), dtype=np.uint64)
    under  = np.zeros((S, G), dtype=np.uint64)
    over   = np.zeros((S, G), dtype=np.uint64)
    eq_neg = np.zeros((S, G), dtype=np.uint64)
    eq_pos = np.zeros((S, G), dtype=np.uint64)

    # Precompute GBM scalars/vectors
    sqrt_dt = np.float32(np.sqrt(DT))
    drift_S = ((MU - 0.5*SIGMAS**2) * DT).astype(np.float32)  # (S,)
    diff_S  = (SIGMAS * sqrt_dt).astype(np.float32)           # (S,)
    one_m_g = (1.0 - GAMMAS).astype(np.float32)               # (G,)
    denom_g = np.log1p(-GAMMAS).astype(np.float32)            # (G,)

    # RNG
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    seed = hash(timestamp) % (2**32)
    rng = np.random.default_rng(seed + worker_idx)

    N = SEEDS_PER_WORKER

    # State
    P = np.full((N, S, G), S0, dtype=np.float32)                 # (B,S,G)
    X = np.full((N, S, G), X0, dtype=np.float32)              # (B,S,G)
    Y = np.full((N, S, G), Y0, dtype=np.float32)              # (B,S,G)

    # Flatten helpers for histogram updates
    SG = S * G
    
    def _accumulate_histogram(theta_3d):
        """
        theta_3d: (B,S,G) with finite entries to be counted; non-finite ignored.
        Updates counts/under/over in-place.
        """
        # reshape to (B, SG)
        V = theta_3d.reshape(N, SG)
        finite_mask = np.isfinite(V)
        if not np.any(finite_mask):
            return

        rows, cols = np.nonzero(finite_mask)     # values positions
        vals = V[rows, cols]

        # Track theta exactly hitting the reflective bounds separately.
        neg_mask = np.isclose(vals, -1.0, atol=1e-7)
        pos_mask = np.isclose(vals,  1.0, atol=1e-7)

        if np.any(neg_mask):
            flat_neg = eq_neg.reshape(SG)
            np.add.at(flat_neg, cols[neg_mask], 1)

        if np.any(pos_mask):
            flat_pos = eq_pos.reshape(SG)
            np.add.at(flat_pos, cols[pos_mask], 1)

        mid_mask = (~neg_mask) & (~pos_mask)
        if not np.any(mid_mask):
            return

        vals = vals[mid_mask]
        cols = cols[mid_mask]

        # bin indices wrt EDGES
        idx = np.searchsorted(EDGES, vals, side='right') - 1

        # under/over per (S,G)
        under_flat = np.zeros(SG, dtype=np.uint64)
        over_flat  = np.zeros(SG, dtype=np.uint64)

        u_mask = idx < 0
        o_mask = idx >= BINS
        m_mask = (~u_mask) & (~o_mask)

        if np.any(u_mask):
            np.add.at(under_flat, cols[u_mask], 1)
        if np.any(o_mask):
            np.add.at(over_flat,  cols[o_mask], 1)

        # mid-bin contributions into (S,G,BINS) via flattened target
        if np.any(m_mask):
            bins_mid = idx[m_mask]             # (K,)
            cols_mid = cols[m_mask]            # (K,)
            flat_counts = counts.reshape(SG, BINS)
            np.add.at(flat_counts, (cols_mid, bins_mid), 1)

        # fold under/over back to (S,G)
        counts_under = under_flat.reshape(S, G)
        counts_over  = over_flat.reshape(S, G)
        under[:] += counts_under
        over[:]  += counts_over

    # Time stepping (only remaining loop)
    for t in range(STEPS):
        # === GBM update for P ===
        z = rng.standard_normal(size=(N, S, G)).astype(np.float32)
        inc = diff_S[None, :, None] * z
        P *= np.exp(inc, dtype=np.float32)  # (N,S,G)

        # === Rebalance X,Y across ALL gammas (vectorized) ===
        # Broadcast P to (B,S,1) once
        P3 = P.copy()

        xg = X  # (N,S,G)
        yg = Y  # (N,S,G)

        x_safe = np.maximum(xg, EPS, dtype=np.float32)  # (N,S,G)
        ratio  = yg / x_safe                       # (N,S,G)
        upper_th = ratio / one_m_g[None, None, :]  # (N,S,G)
        lower_th = ratio * one_m_g[None, None, :]  # (N,S,G)

        upper = P3 > upper_th                      # (N,S,G)
        lower = P3 < lower_th                      # (N,S,G)

        any_adj = upper | lower
        if np.any(any_adj):
            # L only used where an adjustment happens
            L = np.sqrt((xg * yg).astype(np.float32), dtype=np.float32)

            # Upper side: move towards higher P (buy Y, sell X)
            if np.any(upper):
                tmp = np.sqrt(one_m_g[None, None, :] * P3, dtype=np.float32)  # (N,S,G)
                xg[upper] = L[upper] / tmp[upper]
                yg[upper] = L[upper] * tmp[upper]

            # Lower side: move towards lower P (buy X, sell Y)
            if np.any(lower):
                p_safe = np.maximum(P3, EPS, dtype=np.float32)                     # (N,S,G)
                tmp_x  = np.sqrt(one_m_g[None, None, :] / p_safe, dtype=np.float32)  # (N,S,G)
                xg[lower] = L[lower] * tmp_x[lower]

                tmp_y  = np.sqrt(p_safe / one_m_g[None, None, :], dtype=np.float32)  # (N,S,G)
                yg[lower] = L[lower] * tmp_y[lower]
                
            X = xg
            Y = yg

        # === Accumulate theta after warmup ===
        if t >= WARMUP:
            # Pt = Y/X; guard nonpositive
            x_safe2 = np.maximum(X, EPS, dtype=np.float32)
            Pt = Y / x_safe2  # (N,S,G)
            
            # θ = -log(P / Pt) / log(1-γ)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = P3 / Pt
                # Invalidate where denom is 0 or inf
                inv_denom = np.zeros_like(denom_g)
                good_denom = np.isfinite(denom_g) & (denom_g != 0.0)
                inv_denom[good_denom] = 1.0 / denom_g[good_denom]
                theta = -np.log(ratio, dtype=np.float32) * inv_denom[None, None, :]  # (N,S,G)

            _accumulate_histogram(theta)

        # progress tick
        if progress_q is not None and report_every_steps:
            progress_q.put(("step_block", 1))

    return counts, under, over, eq_neg, eq_pos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    S, G = SIGMAS.size, GAMMAS.size
    manager = mp.Manager()
    q = manager.Queue()
    stop_evt = threading.Event()

    bars = {}
    bars['steps'] = tqdm(total=WORKERS * STEPS, desc="Steps progressed", position=1)

    # consumer thread for progress queue
    def consume():
        while not stop_evt.is_set() or not q.empty():
            try:
                typ, val = q.get(timeout=0.1)
                bars['steps'].update(val)
            except queue.Empty:
                pass

    t_cons = threading.Thread(target=consume, daemon=True)
    t_cons.start()

    counts_total = np.zeros((S, G, BINS), dtype=np.uint64)
    under_total  = np.zeros((S, G), dtype=np.uint64)
    over_total   = np.zeros((S, G), dtype=np.uint64)
    eq_neg_total = np.zeros((S, G), dtype=np.uint64)
    eq_pos_total = np.zeros((S, G), dtype=np.uint64)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_worker_chunk, i, q, REPORT_EVERY_STEPS) for i in range(WORKERS)]
        for fut in futs:
            c, u, o, e_neg, e_pos = fut.result()
            counts_total += c
            under_total  += u
            over_total   += o
            eq_neg_total += e_neg
            eq_pos_total += e_pos
            # add the overflow to the last bin
            counts_total[:, :, -1] += o
    stop_evt.set()
    t_cons.join()
    for b in bars.values(): b.close()

    np.savez_compressed(
        OUT_NPZ,
        bin_edges=EDGES.astype(np.float32),
        counts=counts_total,
        underflow=under_total,
        overflow=over_total,
        theta_eq_neg1=eq_neg_total,
        theta_eq_pos1=eq_pos_total,
        sigmas=SIGMAS,
        gammas=GAMMAS,
        meta=np.array([WORKERS, SEEDS_PER_WORKER, STEPS, WARMUP, DT, S0, X0, Y0, MU], dtype=np.float64),
        meta_names=np.array(["workers","seeds_per_worker","steps","warmup","dt","s0","x0","y0","mu"])
    )
    print(f"\nSaved histograms to: {OUT_NPZ} (≪ 1 MB). Time: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
