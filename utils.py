
import numpy as np
import math

TOLERANCE = 1e-6

def is_identity_matrix(A, tol=TOLERANCE):
    return np.allclose(A, np.eye(A.shape[0]), rtol=tol)

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    for i in range(A.shape[1]):
        current_col = A[:, i]
        for j in range(i):
            past_col = A[:, j]
            dot = np.dot(current_col.T, past_col)
            current_col -= dot * past_col
        norm = np.linalg.norm(current_col)
        current_col /= norm
        A[:, i] = current_col
    return A


def verify_gram_schmidt(A: np.ndarray, tol=TOLERANCE) -> bool:
    for i in range(A.shape[1]):
        current_col = A[:, i]
        norm = np.linalg.norm(current_col)
        if not math.isclose(norm, 1, rel_tol=tol):
            print("Norm: ", norm)
            return False
        for j in range(i):
            past_col = A[:, j]
            dot = np.dot(current_col.T, past_col)
            if math.isclose(dot, 0, rel_tol=tol):
                print("Dot: ", dot)
                return False
    return True

def generate_random_matrix(m, n):
    return np.random.randn(m, n)
