import numpy as np

def heron(y, x0, tol = 1e-6, n_max=100000):
  x = x0
  for n in range(n_max):
    x_new = 1/2*x + 1/2*y/x
    if abs(x_new - x) < tol:
      return x_new
    x = x_new
  return x

def fixed_point_iteration(g, x0, N):
  x = x0
  print("n x_n f(x_n) error")
  for n in range(N):
    if x == 0:
      print(f"{n}, undefined, undefined, undefined")
      continue
    print(f"{n}, {x:.5f}, {f(x):.5f}, {abs(x - 2**(1/3)):.5f}")
    x_new = g(x)
    x = x_new
  return x

f = lambda x: x**3 - 2
g1 = lambda x: x - f(x) /3
g2 = lambda x: 2 / (x**2)
g3 = lambda x: x - (x**3 - 2) / (3*x**2)

print("Question One")
print(f"Heron: {heron(2, 1)} , Real: {np.sqrt(2)}")
print(f"Heron: {heron(10, 10)} , Real: {np.sqrt(10)}")
print(f"Heron: {heron(1000, 1000)} , Real: {np.sqrt(1000)}")
print("Implementation: My solution implements Heron's algorithm by iterating in a for loop until the end conditions of max iterations and/or maximum tolerance are reached. Y is the end goal and x0 is the initial x guess")
print("Observations: The difference between the NumPy-computed and the Heron's algorithm-computed values is indistinguishable to 15 digits past the decimal. This 'double' is precise enough for nearly all applications.")
print("")
print("Question Two")
print(f"The root alpha is {2**(1/3):.5f}")
print("\ng1:")
fixed_point_iteration(g1, 1.5, 15)
print("\ng2:")
fixed_point_iteration(g2, 1.5, 15)
print("\ng3:")
fixed_point_iteration(g3, 1.5, 15)