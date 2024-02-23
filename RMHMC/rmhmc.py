import os
import time
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import grad, jit, random, hessian, lax
from jax.scipy.special import expit, logit
from matrixOperations import log_det_A_inv

"""
This code defines a simple example of Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) to estimate the probability parameter of a loaded coin.
"""
# Define the types of coin flips
num_heads = 300
num_tails = 700
y: jnp.ndarray = jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])
alpha: float = 2.0
beta_param: float = 2.0

@jit
def U(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the negative log-posterior for a coin flip problem.

    Args:
    theta: A JAX array representing the log-odds of the probability parameter.

    Returns:
    The negative log-posterior as a scalar.
    """
    p = expit(theta)
    log_likelihood = jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    log_prior = (alpha - 1) * jnp.log(p) + (beta_param - 1) * jnp.log(1 - p)
    return -(log_likelihood + log_prior)

@jit
def G(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Riemannian metric tensor using the Hessian of U(theta).

    Args:
    theta: A JAX array representing the log-odds of the probability parameter.

    Returns:
    The Riemannian metric tensor as a JAX array.
    """
    return hessian(U)(theta)

@jit
def leapfrog(theta: jnp.ndarray, r: jnp.ndarray, eps: float, n_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform leapfrog steps for RMHMC, adjusted for JIT compilation.

    Args:
    theta: Initial theta value.
    r: Initial momentum.
    eps: Step size for the leapfrog integration.
    n_steps: Number of leapfrog steps to perform.

    Returns:
    The final state (theta, r) after leapfrog steps.
    """
    dU_dtheta = grad(U)
    
    def leapfrog_step(i: int, val: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        theta, r = val
        r_half = r - 0.5 * eps * dU_dtheta(theta)
        theta_next = theta + eps * r_half
        r_next = r_half - 0.5 * eps * dU_dtheta(theta_next)
        return theta_next, r_next

    final_state = lax.fori_loop(0, n_steps, leapfrog_step, (theta, r))
    return final_state

def rmhmc(rng_key: jax.random.KeyArray, theta0: jnp.ndarray, eps: float, L: int, num_steps: int) -> jnp.ndarray:
    """
    RMHMC loop for estimating the probability parameter of a loaded coin.

    Args:
    rng_key: Random key for JAX's random number generation.
    theta0: Initial log-odds value for the probability parameter.
    eps: Step size for the leapfrog integration.
    L: Number of leapfrog steps per iteration.
    num_steps: Total number of RMHMC iterations to perform.

    Returns:
    An array of sampled probability parameter values in the original parameter space (0, 1).
    """
    theta = theta0
    samples = []

    for _ in range(num_steps):
        r0 = random.normal(rng_key)
        theta_new, r_new = leapfrog(theta, r0, eps, L)
        current_U = U(theta)
        current_K = 0.5 * jnp.dot(r0, r0)
        new_U = U(theta_new)
        new_K = 0.5 * jnp.dot(r_new, r_new)
        if random.uniform(rng_key) < jnp.exp(current_U - new_U + current_K - new_K):
            theta = theta_new
            samples.append(expit(theta_new))
    
    return jnp.array(samples)

def main() -> None:
    """
    Main function to execute the RMHMC sampling.
    """
    start = time.time()
    seed = int.from_bytes(os.urandom(4), 'big')
    rng_key = random.PRNGKey(seed)
    theta0 = logit(jnp.array(0.5))
    eps = 0.01
    L = 10
    num_steps = 5000
    
    samples = rmhmc(rng_key, theta0, eps, L, num_steps)
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    print(f"Mean estimate of p: {jnp.mean(samples)}")

if __name__ == "__main__":
    main()
