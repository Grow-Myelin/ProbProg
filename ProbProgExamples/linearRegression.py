import numpy as np
import matplotlib.pyplot as plt
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import arviz as az
from typing import Optional

# True parameters for generating synthetic data
true_slope: float = 2.5
true_intercept: float = -1.0
noise_scale: float = 1.0

# Generate synthetic data
np.random.seed(0)
x: np.ndarray = np.linspace(0, 1, 100)
noise: np.ndarray = np.random.normal(scale=noise_scale, size=x.shape)
y: np.ndarray = true_slope * x + true_intercept + noise

def linear_regression_model(x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Bayesian linear regression model using numpyro.
    
    Parameters:
    - x: np.ndarray, predictor variable(s).
    - y: Optional[np.ndarray], response variable. If None, the model performs prediction.
    """
    slope = numpyro.sample('slope', dist.Normal(0, 10))
    intercept = numpyro.sample('intercept', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.Exponential(1.0))
    mean = slope * x + intercept
    with numpyro.plate('data', x.shape[0]):
        numpyro.sample('obs', dist.Normal(mean, sigma), obs=y)

def main():
    # Set up the NUTS kernel and run MCMC
    nuts_kernel = NUTS(linear_regression_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), x=x, y=y)

    # Get the samples and convert to ArviZ InferenceData object
    samples = mcmc.get_samples()
    idata = az.from_numpyro(mcmc)

    # Plot the posterior distributions and save the plot
    az.plot_posterior(idata, var_names=['slope', 'intercept', 'sigma'])
    plt.savefig('plots/linear_regression_posterior_plot.png')

    # Summary of the posterior distribution
    print(az.summary(idata, var_names=['slope', 'intercept', 'sigma']))

if __name__ == "__main__":
    main()