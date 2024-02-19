import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS
from typing import Sequence

def coin_flip_model(observed_data: jnp.ndarray) -> None:
    """
    Bayesian model for coin flipping using a Bernoulli distribution.

    Parameters:
    - observed_data: jnp.ndarray, array of observed coin flips (0s and 1s).
    """
    prob_heads = numpyro.sample("prob_heads", dist.Uniform(0, 1))
    numpyro.sample("obs", dist.Bernoulli(prob_heads), obs=observed_data)

def run_mcmc(model: callable, data: jnp.ndarray, num_warmup: int = 1000, num_samples: int = 1000, num_chains: int = 1) -> MCMC:
    """
    Runs MCMC for a given Bayesian model and observed data.

    Parameters:
    - model: Callable, the Bayesian model function.
    - data: jnp.ndarray, the observed data to fit the model to.
    - num_warmup: int, number of warmup steps.
    - num_samples: int, number of samples to draw.
    - num_chains: int, number of MCMC chains to run.

    Returns:
    - MCMC object after running the MCMC algorithm.
    """
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0), observed_data=data)
    return mcmc

def main():
    # Generate observed data for a biased coin
    key = random.PRNGKey(0)
    prob_heads = 0.7
    num_flips = 100
    observed_data = (random.uniform(key, (num_flips,)) < prob_heads).astype(int)

    # Run MCMC
    mcmc = run_mcmc(coin_flip_model, observed_data)

    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(mcmc)

    # Plotting the posterior
    ax = az.plot_posterior(idata, var_names=["prob_heads"])

    # Save the figure
    plt.savefig("coin_flip_posterior_plot.pdf")

if __name__ == "__main__":
    main()
