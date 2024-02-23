import jax
from typing import Tuple, Callable
import jax.numpy as jnp
from jax import grad, jit,random,lax
from jax.scipy.special import logit, expit
import time
import os

# Example placeholders for y, alpha, beta_param
num_heads = 300
num_tails = 700
# Create an array with 60 heads (1) and 40 tails (0)
y = jnp.concatenate([jnp.ones(num_heads), jnp.zeros(num_tails)])
alpha = 2.0
beta_param = 2.0

@jit
def U(theta):
    """
    Calculate the negative log-posterior for a coin flip problem.

    Args:
    theta: A JAX array representing the log-odds of the probability parameter.

    Returns:
    The negative log-posterior as a scalar.
    """
    p = expit(theta)
    log_likelihood = jnp.sum(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    log_prior = (alpha - 1) * jnp.sum(jnp.log(p)) + (beta_param - 1) * jnp.sum(jnp.log(1 - p)) 
    return - (log_likelihood + log_prior)

@jit
def G(theta):
    """
    Compute the Riemannian metric tensor using the Hessian of U(theta).

    Args:
    theta: A JAX array representing the log-odds of the probability parameter.

    Returns:
    The Riemannian metric tensor as a JAX array.
    """
    return jax.hessian(U)(theta)

@jit
def leapfrog(theta, r, eps, n_steps):
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
    
    def leapfrog_step(i, val):
        theta, r = val
        # Update momentum half step
        r_half = r - 0.5 * eps * dU_dtheta(theta)
        
        # Update position full step
        theta_next = theta + eps * r_half
        
        # Update momentum half step
        r_next = r_half - 0.5 * eps * dU_dtheta(theta_next)
        
        return theta_next, r_next
    
    # Initialize the loop state with the initial values of theta and r
    final_state = lax.fori_loop(0, n_steps, leapfrog_step, (theta, r))
    
    return final_state


def rmhmc(rng_key, theta0, eps, L, num_steps):
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
    
    for i in range(num_steps):
        r0 = random.normal(rng_key)
        theta_new, r_new = leapfrog(theta, r0, eps, L)
        
        # Compute Metropolis-Hastings acceptance (simplified version)
        current_U = U(theta)
        current_K = 0.5 * r0 ** 2
        new_U = U(theta_new)
        new_K = 0.5 * r_new ** 2
        
        # Accept or reject the new state
        if random.uniform(rng_key) < jnp.exp(current_U - new_U + current_K - new_K):
            theta = theta_new
            samples.append(expit(theta_new))  # Store samples in original parameter space (0, 1)
            
    return samples


def main():
    start = time.time()
    seed = int.from_bytes(os.urandom(4), 'big')
    rng_key = random.PRNGKey(seed)
    theta0 = logit(jnp.array(0.5))  # Initial parameter in log-odds space
    eps = 0.01  # Step size
    L = 10  # Number of leapfrog steps
    num_steps = 5000  # Number of RMHMC iterations
    
    samples = rmhmc(rng_key, theta0, eps, L, num_steps)
    end = time.time()
    print(end-start)
    print(f"Mean estimate of p: {jnp.mean(jnp.array(samples))}")

if __name__ == "__main__":
    main()