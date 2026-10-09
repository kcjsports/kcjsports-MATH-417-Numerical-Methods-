import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_eval, x_nodes, y_nodes):
  n = len(x_nodes)
  result = np.zeros_like(x_eval, dtype=float)
  for i in range(n):
    L_i = np.ones_like(x_eval, dtype=float)
    for j in range(n):
      if i != j:
        L_i *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    result += y_nodes[i] * L_i
  return result

def points(n, a=-1, b=1):
    return np.linspace(a, b, n)

g = lambda x: np.sqrt(np.abs(x))
f = lambda x: 1 / (1 + 25 * x**2)

n_values = [5, 8, 20]
x_fine = np.linspace(-1, 1, 500)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

functions = [('f(x) = 1/(1+25x²)', f), ('g(x) = √|x|', g)]

for row, (func_name, func) in enumerate(functions):
    for col, n in enumerate(n_values):
        ax = axes[row, col]
        x_nodes = points(n)
        y_nodes = func(x_nodes)
        y_interp = lagrange_interpolation( x_fine, x_nodes, y_nodes)
        ax.plot(x_fine, func(x_fine), 'b-', linewidth=2, label='Original')
        ax.plot(x_fine, y_interp, 'r--', linewidth=1.5, label='Lagrange interp.')
        ax.scatter(x_nodes, y_nodes, color='black', s=50, zorder=5, label='Nodes')
        ax.set_title(f'{func_name}, n = {n}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        if row == 0:  # f(x)
            ax.set_ylim(-0.5, 2.0)
        else:  # g(x)
            ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

print("Maximum Interpolation Errors:")
print("-" * 50)
for func_name, func in functions:
    print(f"\n{func_name}:")
    for n in n_values:
        x_nodes = points(n)
        y_nodes = func(x_nodes)
        y_interp = lagrange_interpolation(x_fine, x_nodes, y_nodes)
        max_error = np.max(np.abs(func(x_fine) - y_interp))
        print(f"  n = {n:2d}: max error = {max_error:.5f}")