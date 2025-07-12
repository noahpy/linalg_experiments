
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import gram_schmidt


def generate_data_principal_comp(principal_components, principal_values, num_samples):
    """
    Generate a matrix with num_samples, based on principal components and principal values.

    - principal_components: ndarray with size (D, P), where D is the number of dimensions and P is the number of principal components. Each column vector represents a pricipal **normed** vector
    - principal_values: ndarray of principal values with shape (P,1), representing the standard deviation of each principal component
    """
    coefficients = np.random.randn(principal_components.shape[1], num_samples)
    # multiply respective std
    coefficients = np.multiply(coefficients, principal_values)

    M = np.dot(principal_components, coefficients)
    return M

def generate_principal_comps(m, n, max_principal_value=2, min_principal_value=0.1):
    """
    Generate the normed, orthogonal principal components of a matrix with shape (m, n) and also returns a vector of principal values with decreasing order.
    """

    principal_components = np.random.randn(m, n)
    principal_components = gram_schmidt(principal_components)

    principal_values = np.random.uniform(min_principal_value, max_principal_value, n)
    principal_values = np.sort(principal_values)[::-1].reshape(-1, 1)

    return principal_components, principal_values


def check_reconstruction_accuracy(Vh, S, pc, pv, num_samples):
    """
    Given the singular vectors and values, see how accurately they were able to reconstruct the principal components and their standard deviation!
    """

    print("\n\nChecking reconstruction accuracy...")
     
    for i in range(pc.shape[1]):
        mse = np.sum((np.abs(pc[:, i]) - np.abs(Vh[i, :])) ** 2)
        print(f"Mean squared error for principal component {i}: {mse}")

        reconstructed_std = S[i] / math.sqrt(num_samples - 1)
        print(f"Reconstructed standard deviation for principal component {i}: {reconstructed_std}, diff to true: {reconstructed_std - pv[i, 0]}")


if __name__ == "__main__":
    """
    Let us generate a data matrix using given principal components and principal values.
    Let us then use SVD to calculate these principal components back out of the matrix, 
    and lets see how we did!
    """

    np.random.seed(42)

    pc, pv = generate_principal_comps(10, 6)


    print("Generated principal components:")

    print(f"pc = \n{pc}")

    print(f"pv = \n{pv}")
    
    num_samples = 100

    A = generate_data_principal_comp(pc, pv, num_samples).T

    print("\n\nGenerated data matrix with shape: ")

    print(A.shape)


    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    # print(f"U = \n{U}")
    # print(f"S = \n{S}")
    # print(f"Vh = \n{Vh[0]}")

    S_diag = np.diag(S)
    # print(f"S_diag = \n{S_diag}")

    print("\n\nUsing full SVD with shapes:")
    print(f"U.shape = {U.shape}")
    print(f"S_diag.shape = {S_diag.shape}")
    print(f"Vh.shape = {Vh.shape}")

    check_reconstruction_accuracy(Vh, S, pc, pv, num_samples)


    print("\n\nFinally, check reconstruction accuracy for different approximations:")

    dim = len(S)

    dims = []
    errors = []

    for i in range(dim):

        U_approx = U[:, :dim - i]
        S_diag_approx = S_diag[:dim - i, :dim - i]
        Vh_approx = Vh[:dim - i, :]

        print(f"Using {dim - i}-dimensional approximation with shapes:")
        print(f"U_approx.shape = {U_approx.shape}")
        print(f"S_diag_approx.shape = {S_diag_approx.shape}")
        print(f"Vh_approx.shape = {Vh_approx.shape}")

        A_approx = U_approx @ S_diag_approx @ Vh_approx

        diff = A - A_approx

        error = np.linalg.norm(diff, ord='fro')

        print(f"Error: {error}")

        dims.append(dim - i)
        errors.append(error)


    plt.plot(dims, errors)
    plt.xlabel("Dimension")
    plt.ylabel("Forbenius error")
    plt.show()
