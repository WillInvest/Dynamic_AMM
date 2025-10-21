#%%

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

GAMMA = 0.01
SIGMA = 0.2
DT = 1/(365*24*60*5)
X0 = 1_000_000
Y0 = 1_000_000
L0 = np.sqrt(X0 * Y0)
LAM = - np.log(1-GAMMA)
MAX_ITER = 100000000
N = 10000
x = np.linspace(-1, 1, N)[1:-1]
w = np.repeat(1/len(x), len(x))

def k_y_x(y, x):
    mu = x - SIGMA**2 * DT / (2 * LAM)
    nu = SIGMA * np.sqrt(DT) / LAM
    return norm.pdf((y - mu) / nu) / nu

def r_x(x):
    mu = x - SIGMA**2 * DT / (2 * LAM)
    nu = SIGMA * np.sqrt(DT) / LAM
    return 1 - norm.cdf((1 - mu) / nu)

def l_x(x):
    mu = x - SIGMA**2 * DT / (2 * LAM)
    nu = SIGMA * np.sqrt(DT) / LAM
    return norm.cdf((-1 - mu) / nu)

K = np.empty((len(x), len(x)), dtype=float)
for i in range(len(x)):
    # weight on *source* node i
    K[:, i] = k_y_x(x, x[i]) #* w[i]

k_from_minus = k_y_x(x, -1.0)
k_from_plus  = k_y_x(x, +1.0)

ell_vec = np.array([l_x(xi) for xi in x]) * w
r_vec   = np.array([r_x(xi) for xi in x]) * w

ell_minus = l_x(-1.0)
ell_plus  = l_x(+1.0)
r_minus   = r_x(-1.0)
r_plus    = r_x(+1.0)


@dataclass
class Theta_Distribution:
    f: np.ndarray
    mL: float
    mR: float

def main():
    f = np.full(len(x), 0.5)
    mL, mR = 0.0, 0.0
    tol = 1e-8
    
    for _ in range(MAX_ITER):
        # Push forward (raw)
        f_raw = K @ f + mL * k_from_minus + mR * k_from_plus
        mL_raw = float(ell_vec @ f + mL * ell_minus + mR * ell_plus)
        mR_raw = float(r_vec   @ f + mL * r_minus   + mR * r_plus)

        # Raw masses (interior via quadrature)
        m_int_raw = float(np.dot(w, f_raw))
        M_raw = m_int_raw + mL_raw + mR_raw

        # Renormalize (helps kill numerical drift)
        f_new  = f_raw / M_raw
        mL_new = mL_raw / M_raw
        mR_new = mR_raw / M_raw
        
        # L1 distance between consecutive distributions (interior + atoms)
        d_int = float(np.dot(w, np.abs(f_new - f)))
        d_atoms = abs(mL_new - mL) + abs(mR_new - mR)
        L1 = d_int + d_atoms
        
        f, mL, mR = f_new, mL_new, mR_new

        if L1 < tol:
            print(f"L1: {L1}")
            break
        
    return Theta_Distribution(f, mL, mR)

#%%
import matplotlib.pyplot as plt
theta_distribution = main()

f = theta_distribution.f
mL = theta_distribution.mL
mR = theta_distribution.mR
print(f"mL: {mL}, mR: {mR}")
# %%

import numpy as np
import matplotlib.pyplot as plt

mass = f * w  # per-bin probability mass
plt.bar(x, mass, width=w, align='center', edgecolor='k')
plt.xlim(-1, 1)
plt.xlabel('x')
plt.ylabel('f·w (mass per bin)')
plt.show()

# %%
