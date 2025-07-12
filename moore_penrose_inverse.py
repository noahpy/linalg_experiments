
import numpy as np
from utils import generate_random_matrix, is_identity_matrix

ROWS = 100
COLS = 15

A = generate_random_matrix(ROWS, COLS)


def calc_moore_rose_inverse(A):
    """
    Given matrix A, caluclate the moore rose inverse A^+ using SVD.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    S = 1 / S
    S_diag = np.diag(S)
    mr_inverse = Vh.T @ S_diag @ U.T

    return mr_inverse


print("Given random matrix A with shape: ", A.shape)

inv = calc_moore_rose_inverse(A)

print("Calculated Moore Rose inverse with shape: ", inv.shape)

# Apply inverse
if ROWS > COLS:
    out = inv @ A
else:
    out = A @ inv

# Check if it works
if is_identity_matrix(out):
    print("Moore Rose Inverse works!")
else:
    print("Inverse did not work...")
    print("This may be if A is not full rank")
