import numpy as np

def newton_linear(xn, xs, n):
  print("n, x_n, e_n, e_{n+1}/e_n, e_{n+1}/e_n^2") 

  for i in range(n+1):
    en = abs(xn-xs)
    fx = np.exp(xn) - xn - 1
    fx_1 = np.exp(xn) - 1
    xn_1 = xn - (fx) / (fx_1)
    en_1 = abs(xn_1-xs)
    print(f'{i}, {xn:.5f}, {en:.5f}, {abs(en_1 / en):.5f}, {abs(en_1 / (en ** 2)):.5f}')
    xn = xn_1

def newton_quad(xn, xs, n):
  print("n, x_n, e_n, e_{n+1}/e_n, e_{n+1}/e_n^2") 

  for i in range(n+1):
    en = abs(xn-xs)
    fx = np.exp(xn) - xn - 1
    fx_1 = np.exp(xn) - 1
    fx_2 = np.exp(xn)
    xn_1 = xn - ((fx * fx_1) / (fx_1**2 - fx * fx_2))
    en_1 = abs(xn_1-xs)
    print(f'{i}, {xn:.5f}, {en:.5f}, {abs(en_1 / en):.5f}, {abs(en_1 / (en ** 2)):.5f}')
    xn = xn_1

print("Question 1A")
newton_linear(1, 0, 5)
print("The error ratio of en+1/en remains near 0.5, which aligns with linear convergence")

print("Question 1C")
newton_quad(1, 0, 3)
print("The error ratio of en+1/en does not remain constant, which aligns with quadratic convergence. The convergence also happens significantly faster than part A.")


def fixed_point(g, xo, tol = 1e-6, n_max=50):
  xn = xo
  xs = 21 ** (1/3)
  print('n, xn, |xn - xs|')
  for n in range(n_max + 1):
    en = abs(xn - xs)
    print(f'{n}, {xn:.5f}, {en:.5f}')
    if en < tol:
      break
    xn = g(xn)

print("Question 2Aa")
fixed_point(lambda x: (20*x + 21/x**2) / 21, 1)
print("Question 2Ab")
fixed_point(lambda x: x - (x**3 - 21) / (3*x**2), 1)
print("Question 2Ac")
fixed_point(lambda x: x - (x**4 - 21*x) / (x**2 - 21), 1)