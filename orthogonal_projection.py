
import numpy as np
import math


if __name__ == "__main__":

    sqrt2 = math.sqrt(2)
    sqrt6 = math.sqrt(6)

# Basis of subspace in matrix form
    A = np.asarray([[1/sqrt2, 1/sqrt6], [0, -2/sqrt6], [1/sqrt2, -1/sqrt6]])

    print(f"Subspace:\n{A}")

# vector to be projected
    v = np.asarray([3, -1, 4])

# calculate base coefficients
    coefficients = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, v))

# calculate resulting projection
    projection = np.sum(np.multiply(A, coefficients), axis=1)

    print(f"Projection: {projection}")

# calculate residual
    residual = v - projection

    print(f"Residual: {residual}")

# verify residual is orthogonal to the subspace
    print(f"Orthogonality: {np.dot(A.T, residual)}")
