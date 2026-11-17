import numpy as np

def LU_decomp(A):
  a = np.array(A, dtype=float)
  n = A.shape[0]
  M = A.copy()
  for k in range(n-1):
    for i in range(k+1,n):
      m = M[i,k] / M[k,k]
      M[i,k] = m
      for j in range(k+1,n):
        M[i,j] = M[i,j] - m * M[k,j]
  L = np.eye(n) + np.tril(M,-1)
  U = np.triu(M)
  return L, U

def cholesky(A):
  A = np.array(A, dtype=float)
  n = A.shape[0]
  L = np.zeros((n,n))
  for i in range(n):
    for j in range(i):
      s = sum(L[i, k] * L[j, k] for k in range(j))
      L[i, j] = (A[i, j] - s) / L[j, j]
    s = sum(L[i, k] ** 2 for k in range(i))
    L[i, i] = np.sqrt(A[i, i] - s)
  return L

def forward_sub(L,b):
  n = L.shape[0]
  y = np.zeros(n)
  for i in range(n):
    s = sum(L[i, k] * y[k] for k in range(i))
    y[i] = (b[i] - s) / L[i, i]
  return y

def backward_sub(U,y):
  n = U.shape[0]
  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    s = sum(U[i, k] * x[k] for k in range(i + 1, n))
    x[i] = (y[i] - s) / U[i, i]
  return x

A = np.array([[ 5, -5,   0,   0],
              [-5,  7,  -2,   0],
              [ 0, -2,  20, -18],
              [ 0,  0, -18,  19]], dtype=float)

b = np.array([5, -7, 20, -17], dtype=float)

L, U = LU_decomp(A)
g = forward_sub(L, b)
x_lu = backward_sub(U, g)
print("LU solution: x =", x_lu)

Lc = cholesky(A)
y = forward_sub(Lc, b)
x_c = backward_sub(Lc.T, y)
print("Cholesky solution: x =", x_c)