
import numpy as np
import math

TOLERANCE = 1e-6

A = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float)


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


def verify_gram_schmidt(A: np.ndarray) -> bool:
    for i in range(A.shape[1]):
        current_col = A[:, i]
        norm = np.linalg.norm(current_col)
        if not math.isclose(norm, 1, rel_tol=TOLERANCE):
            print("Norm: ", norm)
            return False
        for j in range(i):
            past_col = A[:, j]
            dot = np.dot(current_col.T, past_col)
            if math.isclose(dot, 0, rel_tol=TOLERANCE):
                print("Dot: ", dot)
                return False
    return True


if __name__ == "__main__":

    Q = gram_schmidt(A)

    print("Q:\n", Q)
# print(verify_gram_schmidt(Q))

    R = np.dot(Q.T, A)

    print("R:\n", R)
