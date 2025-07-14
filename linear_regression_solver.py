
import numpy as np
from moore_penrose_inverse import calc_moore_penrose_inverse
import matplotlib.pyplot as plt

def generate_linear_model_data(num_samples, num_covariates, noise_std = 1, max_abs_cov_value = 5, max_abs_bias_value = 5):
    """
    Given the number of sampels and covariates, generate ground truth covariates including bias, then sample points and
    and finally observed data at those sample points added with noise ~ N(0, noise_std^2).

    Args:
        num_samples: int
        num_covariates: int
        noise_std: float
        max_abs_cov_value: float
        max_abs_bias_value: float

    Returns:
        (covariates, sample_points, observed_data)

        - covariates: ndarray with shape (num_covariates + 1, 1)
        - sample_points: ndarray with shape (num_samples, num_covariates + 1)
        - observed_data: ndarray with shape (num_samples, 1)
    """

    # Generate covariates
    covariates = np.random.uniform(-max_abs_cov_value, max_abs_cov_value, (num_covariates, 1))
    bias = np.random.uniform(-max_abs_bias_value, max_abs_bias_value, (1, 1))
    covariates = np.append(covariates, bias, axis=0)

    # Generate sample points
    sample_points = np.random.randn(num_samples, num_covariates + 1)
    sample_points[:, 0] = 1

    # Generate observed data
    observed_data = np.dot(sample_points, covariates) + np.random.normal(0, noise_std, (num_samples, 1))

    return covariates, sample_points, observed_data


def calc_linear_regression_solution(sample_points, observed_data):
    return calc_moore_penrose_inverse(sample_points) @ observed_data

if __name__ == "__main__":
    num_samples = 100
    num_covariates = 5 

    cov, sample_points, data = generate_linear_model_data(num_samples, num_covariates)

    # Calculate linear regression solution via MP-inverse
    solution = calc_linear_regression_solution(sample_points, data)
    
    # print(f"Solution: {solution}")

    # subplot for each covariate, but maximum 9
    num_covariates = min(9, num_covariates)
    num_rows = 3 - (2 - (num_covariates - 1) // 3)
    num_cols = num_covariates % 3 if num_covariates < 3 else 3


    for i in range(num_covariates):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.scatter(sample_points[:, i + 1], data)
        plt.title(f"Cov {i + 1}, Estim: {solution[i+1, 0]:.3f}, True: {cov[i+1, 0]:.3f}")
        sol_vec = np.asarray([solution[0, 0], solution[i + 1, 0]]).reshape(2, 1)
        points_vec = sample_points[:, [0, i + 1]]
        plt.plot(sample_points[:, i + 1], np.dot(points_vec, sol_vec))
    plt.show()
