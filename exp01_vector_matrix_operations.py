"""
Experiment 1 : Vector and Matrix Operations for Deep Learning
Course       : CSA47 - Deep Learning
Libraries    : NumPy, Matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 60)
print(" EXPERIMENT 1 : VECTOR AND MATRIX OPERATIONS")
print("=" * 60)

# ---------------- 1. Vector operations ----------------
a = np.array([3.0, 1.0])
b = np.array([1.0, 3.0])

print("\n--- Vector Operations ---")
print("a =", a, "  b =", b)
print("a + b          :", a + b)
print("a - b          :", a - b)
print("2 * a          :", 2 * a)
print("a * b (element):", a * b)
print("Dot product a.b:", np.dot(a, b))
print("L1 norm of a   :", np.linalg.norm(a, 1))
print("L2 norm of a   :", round(np.linalg.norm(a), 4))
print("Max norm of a  :", np.linalg.norm(a, np.inf))

cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.degrees(np.arccos(cos_sim))
print("Cosine similarity :", round(cos_sim, 4))
print("Angle (degrees)   :", round(angle, 2))

proj_a_on_b = (np.dot(a, b) / np.dot(b, b)) * b
print("Projection of a on b:", proj_a_on_b)

# ---------------- 2. Matrix operations ----------------
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
B = np.array([[1.0, 0.0],
              [2.0, 1.0]])

print("\n--- Matrix Operations ---")
print("A =\n", A)
print("B =\n", B)
print("A @ B (matrix product) =\n", A @ B)
print("A * B (Hadamard product) =\n", A * B)
print("Transpose of B =\n", B.T)
print("Determinant of A :", round(np.linalg.det(A), 4))
A_inv = np.linalg.inv(A)
print("Inverse of A =\n", A_inv)
print("A @ inv(A) (should be I) =\n", A @ A_inv)
print("Rank of A  :", np.linalg.matrix_rank(A))
print("Trace of A :", np.trace(A))

# ---------------- 3. Solving a linear system ----------------
y = np.array([5.0, 10.0])
x = np.linalg.solve(A, y)
print("\nSolve A x = y  ->  x =", x, " check A@x =", A @ x)

# ---------------- 4. Dense layer : Y = XW + b (broadcasting) ----------------
X = np.random.randn(4, 3)          # batch of 4 samples, 3 features
W = np.random.randn(3, 2)          # 3 inputs -> 2 neurons
bias = np.array([0.5, -0.5])
Y = X @ W + bias                   # bias is broadcast to every row
print("\n--- Dense Layer Forward Pass (Y = XW + b) ---")
print("X shape:", X.shape, " W shape:", W.shape, " b shape:", bias.shape)
print("Y shape:", Y.shape)
print("Y =\n", Y)

# ---------------- 5. Visualisation ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

# (a) vectors, their sum and projection
origin = [0, 0]
arrow = dict(angles='xy', scale_units='xy', scale=1)
ax[0].quiver(*origin, *a, color='tab:blue', label='a', **arrow)
ax[0].quiver(*origin, *b, color='tab:orange', label='b', **arrow)
ax[0].quiver(*origin, *(a + b), color='tab:green', label='a + b', **arrow)
ax[0].quiver(*origin, *proj_a_on_b, color='tab:red', width=0.006, label='proj of a on b', **arrow)
ax[0].plot([a[0], proj_a_on_b[0]], [a[1], proj_a_on_b[1]], 'k--', lw=1)
ax[0].set_xlim(-0.5, 4.5); ax[0].set_ylim(-0.5, 4.5)
ax[0].set_aspect('equal'); ax[0].grid(alpha=0.3); ax[0].legend(loc='upper left')
ax[0].set_title("Vector addition and projection")

# (b) linear transformation of unit circle by matrix A
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])
ellipse = A @ circle
ax[1].plot(circle[0], circle[1], 'b--', label='Unit circle')
ax[1].plot(ellipse[0], ellipse[1], 'r', label='A x (transformed)')
ax[1].set_aspect('equal'); ax[1].grid(alpha=0.3); ax[1].legend()
ax[1].set_title("Linear transformation by matrix A")

plt.suptitle("Experiment 1 : Vector and Matrix Operations", fontweight='bold')
plt.tight_layout()
plt.savefig("exp01_output.png", dpi=120)
plt.show()

print("\nPlot saved as exp01_output.png")
