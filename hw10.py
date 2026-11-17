import numpy as np

def gram_schmidt_qr(A):
  A = np.asarray(A, dtype=float)
  m, k = A.shape
  Q = np.zeros((m, k))
  R = np.zeros((k, k))
  for i in range(k):
    w = A[:, i].copy()
    for j in range(i):
      R[j, i] = np.dot(A[:, i], Q[:, j])
      w = w - R[j, i] * Q[:, j]
    R[i, i] = np.linalg.norm(w)
    Q[:, i] = w / R[i, i]
  return Q, R

def solve_least_squares(A, b):
  A = np.asarray(A, dtype=float)
  b = np.asarray(b, dtype=float)
  Q, R = gram_schmidt_qr(A)
  y = Q.T @ b
  n = R.shape[1]
  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
  return x

A = np.array([
  [1, 1, 2, 3],
  [2, 2, 2, 2],
  [4, 3, 2, 2],
  [1, 1, 3, 2],
  [3, 1, 2, 3],
  ], dtype=float)
b = np.array([2, 5, 7, 2, 3], dtype=float)
Q, R = gram_schmidt_qr(A)
y = solve_least_squares(A, b)
print(f"Q =\n{Q}")
print(f"R =\n{R}")
print(f"y =\n{y}")