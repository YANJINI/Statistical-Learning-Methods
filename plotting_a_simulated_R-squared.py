import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time


def plot_r2_simulation(n_sizes, n_simulations, model):
    # Plot the mean and confidence intervals of the simulated sample R-squared.
    #
    # Args:
    #   n_sizes: list or array including different sample sizes
    #   n_simulations: integer representing the number of simulations
    #
    # Returns:
    #   Chart of the simulated R-squared (mean and confidence intervals at each n_sizes)

    r2_q95 = []
    r2_q05 = []
    r2_mean = []

    for s in n_sizes:
        print(s)
        simulated_r2 = [model(s) for _ in range(n_simulations)]
        r2_q95.append(np.quantile(simulated_r2, .95))
        r2_q05.append(np.quantile(simulated_r2, .05))
        r2_mean.append(np.mean(simulated_r2))

    plt.scatter(n_sizes, r2_mean)
    plt.plot(n_sizes, r2_q95)
    plt.plot(n_sizes, r2_q05)
    plt.title("Convergence of simulated R-squared as sample size increases")
    plt.xlabel("Sample size")
    plt.ylabel("Estimated R-squared with its confidence interval")

    if model.__name__ == "_simple_linear_regression_r2":
        plt.savefig("images/simple_regression_r2_convergence.png")
    else:
        plt.savefig("images/multiple_regression_r2_convergence.png")

    plt.show()


def _simple_linear_regression_r2(n):
    # Computes the sample R-squared from a simple linear regression with sample size of n.
    #
    # Args:
    #   n: integer representing the size of sample.
    #
    # Returns:
    #   An integer representing the sample R-squared from a simple linear regression with randomly generated
    #   n pairs of variables from a standard Gaussian distribution.

    x = np.random.randn(n)
    y = 1 + x + np.random.randn(n)
    df = pd.DataFrame({'x': x, 'y': y})
    model = sm.OLS(df['y'], sm.add_constant(df['x'])).fit() # simple linear regression

    return model.rsquared


if __name__ == "__main__":
    n_sizes = np.arange(10, 201, 10)
    reps = 1000

    time0 = time.time()
    plot_r2_simulation(n_sizes=n_sizes, n_simulations=reps, model=_simple_linear_regression_r2)
    time1 = time.time()
    print(f"It took {round(time1 - time0, 2)} seconds")