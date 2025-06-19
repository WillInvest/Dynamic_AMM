# %%
import numpy as np

y0 = 1000000
x0 = 1000000
p0 = y0/x0
gamma = 0.01
L = np.sqrt(y0*x0)
pa = p0/(1-gamma)
pb = p0*(1-gamma)

s1 = 1.0001
if s1 > p0:
    y1 = y0
    x1 = x0
    fee = gamma/(1-gamma) * (y1 - y0)
    ap = (x0 - x1) * s1 - (y1 - y0)/(1-gamma)
elif s1 < pb:
    y1 = L * np.sqrt((s1 / (1-gamma)))
    x1 = L * np.sqrt((1-gamma) / s1)
    fee = gamma/(1-gamma) * (x1 - x0)
    ap = (y0 - y1) - (x1 - x0)/(1-gamma) * s1
else:
    y1 = y0
    x1 = x0
    fee = 0
    ap = 0
    
pv1 = x1 * s1 + y1
tv1 = pv1 + fee
hv1 = x0 * s1 + y0
ng1 = tv1 - hv1

print(f"pv1: {pv1:.8f}, tv1: {tv1:.8f}, hv1: {hv1:.8f}, ng1: {ng1:.8f}, ap1: {ap:.8f}")
print(f"fee: {fee:.8f}, ap: {ap:.8f}")


# %%
# %%
import numpy as np

y0 = 100
x0 = 100
p0 = y0/x0
gamma = 0.1
L = np.sqrt(y0*x0)

s1 = 1.2
if s1 > p0:
    y1 = L * np.sqrt(s1)
    x1 = L / np.sqrt(s1)
    fee = gamma * (x0 - x1)
    ap = (x0 - x1) * (1-gamma) * s1 - (y1 - y0)
elif s1 < p0:
    y1 = L * np.sqrt(s1)
    x1 = L / np.sqrt(s1)
    fee = gamma * (y0 - y1)
    ap = (y0 - y1) * (1-gamma) - (x1 - x0) * s1
else:
    y1 = y0
    x1 = x0
    fee = 0
    ap = 0
    
pv1 = x1 * s1 + y1
tv1 = pv1 + fee
hv1 = x0 * s1 + y0
ng1 = tv1 - hv1

print(f"pv1: {pv1:.4f}, tv1: {tv1:.4f}, hv1: {hv1:.4f}, ng1: {ng1:.4f}, ap1: {ap:.4f}")
print(f"fee: {fee:.4f}, ap: {ap:.4f}")


# %%

# %%
# %%