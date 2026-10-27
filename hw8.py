import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return np.array([
        -x[1] - x[2],
        x[0] + a * x[1],
        b + x[2] * (x[0] - c)
    ])

def fz(t, x):
    return np.array([
        [0.0, -1.0, -1.0],
        [1.0, a, 0.0],
        [x[2], 0.0, x[0] - c]
    ])

def forward_euler(y0, t0, T, h):
    N = int(round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.empty((N + 1, 3))
    Y[0] = y0
    for n in range(N):
        Y[n + 1] = Y[n] + h * f(t[n], Y[n])
    return t, Y

def backward_euler(y0, t0, T, h, tol=1e-10, maxit=100):
    N = int(round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.empty((N + 1, 3))
    Y[0] = y0
    I = np.eye(3)
    for n in range(N):
        z = Y[n].copy()
        for _ in range(maxit):
            G = Y[n] + h * f(t[n + 1], z) - z
            Gz = h * fz(t[n + 1], z) - I
            dz = np.linalg.solve(Gz, -G)
            z = z + dz
            if np.linalg.norm(G) < tol:
                break
        Y[n + 1] = z
    return t, Y
    
a = 0.1
b= 0.1
c= 14.0
y0 = np.array([10.0, 10.0, 0.0])
t0 = 0
T = 40
h = 0.02
tf, Yf = forward_euler(y0, t0, T, h)
tb, Yb = backward_euler(y0, t0, T, h)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(Yf[:, 0], Yf[:, 1], lw=0.5)
axes[0, 0].set_title(f"Forward Euler, xy-plot (h={h})")
axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
axes[0, 1].plot(Yf[:, 0], Yf[:, 2], lw=0.5)
axes[0, 1].set_title(f"Forward Euler, xz-plot (h={h})")
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
axes[1, 0].plot(Yb[:, 0], Yb[:, 1], lw=0.5)
axes[1, 0].set_title(f"Backward Euler, xy-plot (h={h})")
axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
axes[1, 1].plot(Yb[:, 0], Yb[:, 2], lw=0.5)
axes[1, 1].set_title(f"Backward Euler, xz-plot (h={h})")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("z")
plt.tight_layout()
plt.show()