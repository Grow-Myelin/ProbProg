
import jax
import jax.numpy as jnp
from jax import random, grad,jit
import matplotlib.pyplot as plt
from jax.scipy.stats import beta, bernoulli
from functools import partial

def log_posterior(
    p: float,
    alpha: float,
    beta_param: float,
    heads: int,
    tails: int
) -> float:
    """Calculates the log-posterior for Bayesian coin flip model.

    Args:
        p: Probability of heads (between 0 and 1).
        alpha: Alpha parameter of beta prior distribution.
        beta_param: Beta parameter of beta prior distribution.
        heads: Number of observed heads.
        tails: Number of observed tails.

    Returns:
         The log-posterior value.
    """
    log_heads = jnp.log(heads)
    log_tails = jnp.log(tails)
    log_prior = jnp.sum(jnp.log(jax.scipy.stats.beta.pdf(p, alpha, beta_param)))  
    log_likelihood = log_heads * jnp.log(p) + log_tails * jnp.log(1 - p)
    return jnp.sum(log_prior + log_likelihood) 

def potential_fn(params: dict) -> float:
    """Calculates the negative log-posterior (potential function).

    Args:
        params: A dictionary containing the 'p' parameter.

    Returns:
        The negative log-posterior value.
    """
    p = params['p']  # Extract the 'p' parameter from the dictionary
    return -log_posterior(p, alpha, beta_prior, heads, tails)  

def grad_potential_fn(params: dict) -> float:
    """Calculates the gradient of the negative log-posterior.

    Args:
        params: A dictionary containing the 'p' parameter.

    Returns:
        The gradient of the negative log-posterior.
    """
    p = params['p']
    return jnp.array(grad(lambda x: -log_posterior(x, alpha, beta_prior, heads, tails))(p))

log_posterior_jit = jit(log_posterior)  
grad_log_posterior_jit = jit(grad(log_posterior, argnums=0))

# Leapfrog integrator for simulating Hamiltonian dynamics
def leapfrog(
    grad_log_prob: callable,
    position: float,
    momentum: float, 
    step_size: float, 
    num_steps: int,
    alpha: float,
    beta_param: float,
    heads: int,
    tails: int
) -> tuple:
    """Implements the leapfrog integrator for Hamiltonian dynamics.

    Args:
        grad_log_prob:  A callable function to calculate the gradient of the log probability.
        position: Current position (parameter value).
        momentum: Current momentum.
        step_size: Step size for the integrator.
        num_steps: Number of leapfrog steps to take.
        alpha: Alpha parameter of the beta prior.
        beta_param: Beta parameter of the beta prior.
        heads: Number of observed heads.
        tails: Number of observed tails.

    Returns:
        A tuple containing the new position and new momentum.
    """
    grad_log_prob_with_params = partial(grad_log_prob, alpha=alpha, beta_param=beta_param, heads=heads, tails=tails)
    for _ in range(num_steps):
        grad_position = grad_log_prob_with_params(position)
        momentum = momentum - step_size * grad_position  # Update momentum
        position = position + step_size * momentum  # Update position
    return position, momentum

# Hamiltonian Monte Carlo step
def hmc_step(
    rng_key: jnp.ndarray,
    position: float, 
    step_size: float, 
    num_leapfrog_steps: int,
    alpha: float, 
    beta_param: float, 
    heads: int,
    tails: int
) -> tuple:
    """Performs a single Hamiltonian Monte Carlo (HMC) step.

    Args:
        rng_key: JAX PRNGKey for random number generation.
        position: Current position (parameter value).
        step_size: Step size for the leapfrog integrator.
        num_leapfrog_steps: Number of leapfrog steps to take.
        alpha: Alpha parameter of the beta prior.
        beta_param: Beta parameter of the beta prior.
        heads: Number of observed heads.
        tails: Number of observed tails.

    Returns:
        A tuple containing the new position and whether the step was accepted.
    """
    momentum = random.normal(rng_key, shape=(1,))
    current_H = -log_posterior(position, alpha, beta_param, heads, tails)
    new_position, new_momentum = leapfrog(grad_log_posterior_jit, position, momentum, step_size, num_leapfrog_steps, alpha, beta_param, heads, tails)
    new_H = -log_posterior(new_position, alpha, beta_param, heads, tails)
    accept = random.uniform(rng_key) < jnp.exp(current_H - new_H)
    return jnp.where(accept, new_position, position), accept


def get_hmc_config() -> dict:
    """Returns a dictionary containing the HMC configuration parameters."""
    return {
        "alpha": 1,              # Alpha parameter for beta prior
        "beta_prior": 1,         # Beta parameter for beta prior 
        "heads": 40,             # Number of observed heads
        "tails": 20,             # Number of observed tails
        "num_burnin": 500,       # Number of burn-in samples
        "num_samples": 5500,     # Total number of samples (including burn-in)
        "step_size": 0.05,       # HMC step size
        "num_leapfrog_steps": 30 # Number of leapfrog steps
    }


def main():
    """Executes the HMC sampling procedure."""
    config = get_hmc_config()

    rng_key = random.PRNGKey(0)
    samples = []
    position = 0.5  # Initial position

    for _ in range(config["num_samples"]):
        rng_key, subkey = random.split(rng_key)
        position, _ = hmc_step(
            subkey, 
            position, 
            config["step_size"], 
            config["num_leapfrog_steps"], 
            config["alpha"],
            config["beta_prior"], 
            config["heads"], 
            config["tails"]
        )
        samples.append(position)

    samples = samples[config["num_burnin"]:]
    print(f"Estimated probability of heads: {jnp.mean(jnp.array(samples))}")

if __name__ == "__main__":
    main()