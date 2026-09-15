import numpy as np

def newton_iteration(A, x0, lam0, tol=1e-12, max_iter=50):
    n = len(x0)
    x = x0 / np.linalg.norm(x0)  # Normalize initial guess
    lam = lam0
    
    for iteration in range(max_iter):
        # Compute F(x, λ)
        F_upper = A @ x - lam * x  # 3x1
        F_lower = np.dot(x, x) - 1  # scalar
        F = np.hstack([F_upper, F_lower])  # 4x1
        
        # Check convergence
        residual = np.linalg.norm(F)
        if residual < tol:
            print(f"  Converged in {iteration} iterations")
            return lam, x, True
        
        # Build Jacobian
        J = np.zeros((n+1, n+1))
        J[:n, :n] = A - lam * np.eye(n)  # A - λI
        J[:n, n] = -x                      # -x^T
        J[n, :n] = 2 * x                   # 2x
        J[n, n] = 0                        # 0
        
        # Solve J * [Δx; Δλ] = -F
        delta = np.linalg.solve(J, -F)
        
        # Update
        x = x + delta[:n]
        lam = lam + delta[n]
    
    print(f"  Did not converge after {max_iter} iterations")
    return lam, x, False


print("\n**Question 1B:**\n")

# Define matrix A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

# Try different initial guesses to find all eigenvalues ([x1,x2,x3], lam)
initial_guesses = [
    (np.array([1, 0, 0]), 0.5),
    (np.array([0, 1, 0]), -1),
    (np.array([1, 1, 0]), 9),
    (np.array([1, 1, 1]), 13),
    (np.array([1, -2, 1]), 0),
    (np.array([0, 0, 1]), 1),
]

found_eigenvalues = []
found_eigenvectors = []

for i, (x0, lam0) in enumerate(initial_guesses):
    print(f"Initial guess {i+1}: x0 = {x0}, λ0 = {lam0}")
    lam, x, converged = newton_iteration(A, x0, lam0)
    
    if converged:
        is_new = True
        for ev in found_eigenvalues:
            if abs(lam - ev) < 1e-6:
                is_new = False
                break
        
        if is_new:
            found_eigenvalues.append(lam)
            found_eigenvectors.append(x)
            print(f"  λ = {lam:.5f}")
        else:
            print(f"  λ = {lam:.5f} (already found)")

# Display results
print("\nNewton Iteration:")
for i, (lam, x) in enumerate(zip(found_eigenvalues, found_eigenvectors)):
    print(f"λ{i+1} = {lam:.5f}")
    print(f"x{i+1} = [{x[0]:.5f}, {x[1]:.5f}, {x[2]:.5f}]\n")

# Compare with numpy's eigenvalue computation
eigenvalues_np, eigenvectors_np = np.linalg.eig(A)
eigenvalues_np = eigenvalues_np[::-1]

print("Numpy:")
for i, ev in enumerate(eigenvalues_np):
    print(f"  λ{i+1} = {ev:.5f}")


print("\n**Question 3C:**\n")

def F(z):
    x1, x2, lam = z
    F1 = 2*x1 + lam*(3*x1**2 - 1)
    F2 = 2*x2 - 1 - 2*lam*x2
    F3 = x1**3 - x1 + 0.5 - x2**2
    return np.array([F1, F2, F3])

def Jacobian(z):
    x1, x2, lam = z
    J = np.array([
        [2 + 6*lam*x1, 0, 3*x1**2 - 1],
        [0, 2 - 2*lam, -2*x2],
        [3*x1**2 - 1, -2*x2, 0]
    ])
    return J

def newton_opt(z0, tol=1e-10, max_iter=50):
    z = np.array(z0)
    
    print(f"{'Iter'} {'x1'} {'x2'} {'lambda'} {'||F||'}")    
    for k in range(max_iter):
        Fz = F(z)
        norm_F = np.linalg.norm(Fz)
        
        print(f"{k:} {z[0]:.5f} {z[1]:.5f} {z[2]:.5f} {norm_F:<.2e}")
        
        if norm_F < tol:
            print(f"\nConverged after {k} iterations!")
            return z
        
        J = Jacobian(z)
        delta_z = np.linalg.solve(J, -Fz)
        z = z + delta_z
    
    print("Max iterations reached")
    return z

# Initial guess
z0 = [0.5, 0.7, 0.5]

print("Newton's Optimization")
print(f"\nInitial guess: x1 = {z0[0]}, x2 = {z0[1]}, lambda = {z0[2]}\n")

# Run Newton's method
solution = newton_opt(z0)

x1_star, x2_star, lambda_star = solution

print("SOLUTION")
print(f"x* = ({x1_star:.5f}, {x2_star:.5f})")
print(f"λ* = {lambda_star:.5f}")