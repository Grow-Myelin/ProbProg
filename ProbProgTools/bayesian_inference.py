import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from numpyro.infer import MCMC, NUTS

import data_initialization as DI


def plot_posteriors(posterior_samples: Dict[str, jnp.ndarray], industry: str) -> None:
    """
    Plots the posterior distributions for a given industry.

    Args:
        posterior_samples: Samples from the posterior distribution as a dictionary where keys are parameter names.
        industry (str): The name of the industry for which the posterior distributions are plotted.
    """
    plt.figure(figsize=(12, 10))
    num_bins = 30
    params_to_plot = ['mean1', 'mean2', 'sigma1', 'sigma2']
    for i, param in enumerate(params_to_plot):
        plt.subplot(3, 2, i + 1)
        sns.histplot(posterior_samples[param], bins=num_bins, kde=True)
        plt.title(f'Posterior of {param} for {industry}')

    if 'mix' in posterior_samples:
        mix_samples = posterior_samples['mix']
        for i in range(mix_samples.shape[1]):  # Number of components in 'mix'
            plt.subplot(3, 2, len(params_to_plot) + i + 1)
            sns.histplot(mix_samples[:, i], bins=num_bins, kde=True)
            plt.title(f'Posterior of mix[{i}] for {industry}')

    plt.tight_layout()
    plt.savefig(f'plots/posteriors_{industry}.png')


def run_bayesian_inference(simulator: DI.StockMarketSimulator, n_samples: int = 500, n_warmup: int = 100) -> None:
    """
    Runs Bayesian inference for each industry and plots the posterior distributions.

    Args:
        simulator: An instance of StockMarketSimulator containing stock prices and industry mappings.
        n_samples (int): Number of samples to draw from the posterior distribution.
        n_warmup (int): Number of warmup steps for the sampler.
    """
    def stock_price_model(data: jnp.ndarray) -> None:
        mix = numpyro.sample('mix', dist.Dirichlet(jnp.array([1., 1.])))
        mean1 = numpyro.sample('mean1', dist.Normal(0, 10))
        mean2 = numpyro.sample('mean2', dist.Normal(0, 10))
        sigma1 = numpyro.sample('sigma1', dist.HalfCauchy(5))
        sigma2 = numpyro.sample('sigma2', dist.HalfCauchy(5))

        with numpyro.plate('data', len(data)):
            assignment = numpyro.sample('assignment', dist.Categorical(mix))
            means = jnp.array([mean1, mean2])[assignment]
            sigmas = jnp.array([sigma1, sigma2])[assignment]
            numpyro.sample('obs', dist.Normal(means, sigmas), obs=data)

    for industry in set(simulator.industry_map.values()):
        industry_stocks = [stock for stock, ind in simulator.industry_map.items() if ind == industry]
        industry_data = simulator.stock_prices[industry_stocks].values.flatten()

        nuts_kernel = NUTS(stock_price_model)
        mcmc = MCMC(nuts_kernel, num_samples=n_samples, num_warmup=n_warmup)
        seed = int.from_bytes(os.urandom(4), 'big')
        key = jax.random.PRNGKey(seed)  # Initialize JAX PRNG key
        mcmc.run(key, industry_data)
        mcmc.print_summary()

        posterior_samples = mcmc.get_samples()
        plot_posteriors(posterior_samples, industry)


def main() -> None:
    """
    Main function to initialize the simulator and run Bayesian inference.
    """
    simulator = DI.StockMarketSimulator()
    run_bayesian_inference(simulator)


if __name__ == "__main__":
    main
