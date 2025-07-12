
import numpy as np
from utils import gram_schmidt

A = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float)

if __name__ == "__main__":

    Q = gram_schmidt(A)

    print("Q:\n", Q)
# print(verify_gram_schmidt(Q))

    R = np.dot(Q.T, A)

    print("R:\n", R)
