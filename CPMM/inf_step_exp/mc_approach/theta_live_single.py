"""Interactive single-path theta simulation with live plotting."""

import math
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- Tunables ---
STEPS   = 10_000
WARMUP  = 500
DT      = 1.0 / (365.0 * 24.0 * 60 * 5)
MU      = 0.0
SIGMA   = 0.20
GAMMA   = 0.005
S0      = 1.0
X0, Y0  = 1_000_000.0, 1_000_000.0
BINS    = 128
BMIN    = -1.0
BMAX    = 1.0
EPS     = 1e-8

_ANIM = True


@dataclass
class SimulationState:
    """Container for the evolving AMM path."""

    price: float
    reserve_x: float
    reserve_y: float
    timestamps: list
    prices: list


def main() -> None:
    rng = np.random.default_rng(int(time.time()))

    diff = SIGMA * math.sqrt(DT)
    drift = (MU - 0.5 * SIGMA * SIGMA) * DT
    denom = np.log1p(-GAMMA)
    inv_denom = 1.0 / denom if denom != 0.0 else 0.0

    edges = np.linspace(BMIN, BMAX, BINS + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    theta_counts = np.zeros(BINS, dtype=np.int64)

    theta_hits_neg1 = 0
    theta_hits_pos1 = 0
    underflow = 0
    overflow = 0

    state = SimulationState(
        price=S0,
        reserve_x=X0,
        reserve_y=Y0,
        timestamps=[0.0],
        prices=[S0],
    )

    fig, (ax_price, ax_theta) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    plt.tight_layout(pad=2.0)

    (line_price,) = ax_price.plot([], [], lw=1.8, color="tab:blue")
    ax_price.set_title("Price Path")
    ax_price.set_xlabel("Step")
    ax_price.set_ylabel("S_t")

    bars = ax_theta.bar(centers, np.zeros_like(centers), width=widths, align="center", color="tab:orange")
    ax_theta.set_title("Theta Distribution (normalized)")
    ax_theta.set_xlabel("Theta")
    ax_theta.set_ylabel("Probability")
    ax_theta.set_ylim(0.0, 0.15)

    text_stats = ax_theta.text(0.02, 0.95, "", transform=ax_theta.transAxes, va="top", ha="left", fontsize=10)

    def update(step: int):
        nonlocal theta_counts, theta_hits_neg1, theta_hits_pos1, underflow, overflow
        price = state.price
        x_res = state.reserve_x
        y_res = state.reserve_y

        z = rng.standard_normal()
        step_factor = math.exp(drift + diff * z)
        price *= step_factor

        ratio = y_res / max(x_res, EPS)
        upper_th = ratio / (1.0 - GAMMA)
        lower_th = ratio * (1.0 - GAMMA)

        if price > upper_th or price < lower_th:
            L = math.sqrt(x_res * y_res)
            price_safe = max(price, EPS)
            if price > upper_th:
                tmp = math.sqrt((1.0 - GAMMA) * price)
                x_res = L / tmp
                y_res = L * tmp
            else:
                tmp_x = math.sqrt((1.0 - GAMMA) / price_safe)
                tmp_y = math.sqrt(price_safe / (1.0 - GAMMA))
                x_res = L * tmp_x
                y_res = L * tmp_y

        state.price = price
        state.reserve_x = x_res
        state.reserve_y = y_res

        state.timestamps.append(step + 1)
        state.prices.append(price)

        if step >= WARMUP:
            Px = max(x_res, EPS)
            Pt = y_res / Px
            ratio_pt = price / max(Pt, EPS)
            theta = -math.log(ratio_pt) * inv_denom if inv_denom != 0.0 else float("nan")
            theta = np.clip(theta, BMIN, BMAX)

            if np.isfinite(theta):
                if np.isclose(theta, -1.0, atol=1e-7):
                    theta_hits_neg1 += 1
                elif np.isclose(theta, 1.0, atol=1e-7):
                    theta_hits_pos1 += 1

                bin_idx = np.searchsorted(edges, theta, side="right") - 1
                if bin_idx < 0:
                    underflow += 1
                elif bin_idx >= BINS:
                    overflow += 1
                    theta_counts[-1] += 1
                else:
                    theta_counts[bin_idx] += 1

        line_price.set_data(state.timestamps, state.prices)
        ax_price.set_xlim(0, max(state.timestamps[-1], 1))

        y_min = min(state.prices)
        y_max = max(state.prices)
        if y_min == y_max:
            y_padding = y_min * 0.05 if y_min != 0 else 1.0
            ax_price.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            ax_price.set_ylim(y_min * 0.98, y_max * 1.02)

        total = theta_counts.sum()
        heights = theta_counts / total if total > 0 else np.zeros_like(theta_counts, dtype=np.float64)
        for bar, height in zip(bars, heights):
            bar.set_height(height)

        if total > 0 and heights.max() > ax_theta.get_ylim()[1]:
            ax_theta.set_ylim(0.0, min(1.0, heights.max() * 1.2))

        text_stats.set_text(
            f"samples={total}\nθ=-1 hits={theta_hits_neg1}\nθ=+1 hits={theta_hits_pos1}\nunder={underflow} over={overflow}"
        )

        return [line_price, *bars, text_stats]

    def init():
        line_price.set_data([], [])
        for bar in bars:
            bar.set_height(0.0)
        text_stats.set_text("")
        return [line_price, *bars, text_stats]

    global _ANIM
    _ANIM = FuncAnimation(
        fig,
        update,
        frames=range(STEPS),
        init_func=init,
        interval=30,
        blit=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
