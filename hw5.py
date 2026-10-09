import numpy as np
import matplotlib.pyplot as plt

def chebyshev(n):
    i = np.arange(n)
    return np.cos((2*i + 1) * np.pi / (2*n))

def lagrange(func, n):
    x = chebyshev(n)
    # print(x)
    y = func(x)
    def p(xx):
        xx = np.atleast_1d(np.asarray(xx, dtype=float))
        total = np.zeros_like(xx)
        for j in range(n):
            Lj = np.ones_like(xx)
            for m in range(n):
                if m != j:
                    Lj *= (xx - x[m]) / (x[j] - x[m])
            total += y[j] * Lj
        return total
    return p

f = lambda x: 1 / (1 + 25 * x**2)
g = lambda x: np.sqrt(np.abs(x))

xx = np.linspace(-1, 1, 1000)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (eqn, name) in zip(axes, [(f, "f(x)=1/(1+25x^2)"), (g, "g(x)=sqrt(|x|)")]):
    ax.plot(xx, eqn(xx), 'k', lw=2, label=name)
    for n in [5, 8, 20]:
        p = lagrange(eqn, n)
        ax.plot(xx, p(xx), label=f"interp n={n}")
    ax.legend()
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.grid(True)
fig.tight_layout()
plt.show()