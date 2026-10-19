import numpy as np


f = lambda x: 4 / (1 + x**2)


def sum_3point_gauss_quad(K, a=0, b=1, fun=f):
    approx = 0.0
    h = (b - a) / K
    half = h / 2
    node = np.sqrt(3/5)
    for j in range(K):
        mid = a + (j + 0.5) * h
        x1 = mid - half * node
        x2 = mid
        x3 = mid + half * node
        approx += (5*fun(x1) + 8*fun(x2) + 5*fun(x3)) / (18*K)
    return approx


I_exact = np.pi
Ks = [1, 2, 4, 6, 8, 16, 32, 64]
vals = {}
errs = {}

print("Part a")
print(f"{'K':>4}  {'I(K)':>18}  {'error':>12}")
print("-" * 40)
for K in Ks:
    val = sum_3point_gauss_quad(K)
    vals[K] = val
    errs[K] = abs(val - I_exact)
    print(f"{K:4d}  {val:18.15f}  {errs[K]:12.3e}")


print("Part b")
print(f"{'k':>4}  {'alpha':>10}")
print("-" * 18)
for K in Ks:
    K2 = 2 * K
    if K2 in vals:
        e1 = errs[K]
        e2 = errs[K2]
        if e2 > 0 and e1 > 0:
            alpha = np.log(e1 / e2) / np.log(2)
            print(f"{K:4d}  {alpha:10.4f}")
        else:
            print(f"{K:4d}  {'--- (round-off floor)':>10}")