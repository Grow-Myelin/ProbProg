import os
import time
from typing import Any,Tuple, Callable, List
import jax.numpy as jnp
from jax import jit, grad, random, lax, jacrev
from jax.scipy.linalg import solve
import matplotlib.pyplot as plt


def lorenz_system(state: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray:
    """
    Defines the Lorenz system of differential equations.

    Args:
        state: A numpy array containing the current state (x, y, z).
        t: The current time (unused as the Lorenz system is time-independent).
        params: A numpy array containing the parameters of the system (sigma, rho, beta).

    Returns:
        The derivatives of the system as a numpy array.
    """
    sigma, rho, beta = params
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return jnp.array([dx_dt, dy_dt, dz_dt])

def harmonic_oscillator_system(state: jnp.ndarray, t: float, params: jnp.ndarray) -> jnp.ndarray:
    """
    Defines the harmonic oscillator system of differential equations.

    Args:
        state: A numpy array containing the current state (x, v).
        t: The current time (unused as the system is time-independent).
        params: A numpy array containing the parameters of the system (k).

    Returns:
        The derivatives of the system as a numpy array.
    """
    k, = params
    x, v = state
    dx_dt = v
    dv_dt = -k * x
    return jnp.array([dx_dt, dv_dt])


############################################
system = harmonic_oscillator_system
############################################
jacobian_system = jacrev(system, argnums=0)

@jit
def semi_implicit_euler_step(state: jnp.ndarray, t: float, dt: float, params: jnp.ndarray) -> jnp.ndarray:
    """
    Performs a single semi-implicit Euler step.

    Args:
        state: The current state of the system.
        t: The current time.
        dt: The time step to use for the Euler step.
        params: The parameters of the system.

    Returns:
        The updated state after the Euler step.
    """
    f = system(state, t, params)
    J = jacobian_system(state, t, params)
    delta = solve(jnp.eye(len(state)) - dt * J, f)
    return state + dt * delta

def simulate_system(initial_state: jnp.ndarray, dt: float, num_steps: int, params: jnp.ndarray) -> jnp.ndarray:
    """
    Simulates the system over a specified number of steps.

    Args:
        initial_state: The initial state of the system.
        dt: The time step to use for the simulation.
        num_steps: The number of steps to simulate.
        params: The parameters of the system.

    Returns:
        A numpy array containing the state of the system at each time step.
    """
    def body_fun(i: int, states: jnp.ndarray) -> jnp.ndarray:
        states = states.at[i + 1].set(semi_implicit_euler_step(states[i], i * dt, dt, params))
        return states

    states = jnp.zeros((num_steps + 1, len(initial_state)))
    states = states.at[0].set(initial_state)
    return lax.fori_loop(0, num_steps, body_fun, states)

# Parameters for the Lorenz system
start = time.time()

############################################
# Simulation parameters
# sigma = 10.0
# rho = 28.0
# beta = 8.0 / 3.0
# param_names = ['sigma', 'rho', 'beta']
# params = jnp.array([sigma, rho, beta])
param_names = ['k']
params = jnp.array([1.0])
# initial_state = jnp.array([10.0, 10.0, 10.0])  # Modified initial conditions
initial_state = jnp.array([1.0, 0.0])  # Modified initial conditions
n_steps = 1000 # Adjust as needed
############################################


dt = 0.0001
num_steps = 1000 # Increased num_steps

# Run the simulation 
states = simulate_system(initial_state, dt, num_steps, params)
@jit
def U(params: jnp.ndarray, states: jnp.ndarray) -> float:
    """
    Calculates the negative log-posterior for a given parameter set and observed states.

    Args:
        params: A numpy array containing the parameters for the simulation.
        states: A numpy array of observed states to compare against the simulation.

    Returns:
        The negative log-posterior value as a float.
    """
    simulated_states = simulate_system(initial_state, dt, num_steps, params)
    log_likelihood = -0.5 * jnp.sum((simulated_states - states)**2)
    log_prior = -jnp.sum(jnp.log(params))  # Simple uniform prior, adjust if needed
    return - (log_likelihood + log_prior)

@jit
def leapfrog(params: jnp.ndarray, r: jnp.ndarray, eps: float, n_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs the leapfrog integration for Hamiltonian dynamics.

    Args:
        params: The initial parameters for the leapfrog steps.
        r: The initial momentum for the leapfrog steps.
        eps: The step size for the integration.
        n_steps: The number of leapfrog steps to perform.

    Returns:
        A tuple containing the updated parameters and momenta after the leapfrog steps.
    """
    dU_dparams = grad(U, argnums=0)

    def leapfrog_step(i: int, val: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        params, r = val
        r_half = r - 0.5 * eps * dU_dparams(params, states)
        params_next = params + eps * r_half
        r_next = r_half - 0.5 * eps * dU_dparams(params_next, states)
        return params_next, r_next

    return lax.fori_loop(0, n_steps, leapfrog_step, (params, r))

def RMHMC_Step(rng_key: jnp.ndarray, params: jnp.ndarray, eps: float, L: int, states: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    """
    Performs a single step of the Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) algorithm.

    Args:
        rng_key: A random key for JAX's random number generation.
        params: The current parameters of the model.
        eps: The step size for the leapfrog integration.
        L: The number of leapfrog steps to perform.
        states: The observed data.

    Returns:
        A tuple containing the new parameters after the RMHMC step and a boolean indicating whether the step was accepted.
    """
    r0 = random.normal(rng_key, shape=params.shape)  # Draw random momentum
    params_new, r_new = leapfrog(params, r0, eps, L)

    current_U = U(params, states)
    current_K = 0.5 * jnp.dot(r0, r0)
    new_U = U(params_new, states)
    new_K = 0.5 * jnp.dot(r_new, r_new)

    log_accept_ratio = current_U - new_U + current_K - new_K
    accept = random.uniform(rng_key) < jnp.exp(log_accept_ratio)

    return lax.cond(accept, lambda: (params_new, True), lambda: (params, False))

@jit
def rmhmc(rng_key: jnp.ndarray, theta0: jnp.ndarray, eps: float, L: int, states: jnp.ndarray) -> jnp.ndarray:
    """
    Runs the Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) algorithm for parameter estimation.

    Args:
        rng_key: A random key for JAX's random number generation.
        theta0: The initial parameters for the RMHMC algorithm.
        eps: The step size for the leapfrog integration.
        L: The number of leapfrog steps to perform.
        states: The observed data.

    Returns:
        A numpy array containing the samples of parameters generated by the RMHMC algorithm.
    """
    def body_fun(i: int, val: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key, params, samples = val
        key, subkey = random.split(key)
        params, accepted = RMHMC_Step(subkey, params, eps, L, states)
        samples = samples.at[i].set(params)
        return key, params, samples

    samples = jnp.zeros((n_steps, theta0.size))  # Adjust the shape as necessary
    _, _, samples = lax.fori_loop(0, n_steps, body_fun, (rng_key, theta0, samples))
    return samples

def main():
    start = time.time()
    seed = int.from_bytes(os.urandom(4), 'big')
    rng_key = random.PRNGKey(seed)

    # Parameter initialization
    # theta0 = jnp.array([12.0, 26.0, 2.0])
    ##############################
    theta0 = jnp.array([.50])
    ##############################
    #theta0 = jnp.array([5.0,7.0,1.0])
    eps = .01 # Step size
    L = 50     # Number of leapfrog steps

    # Run RMHMC sampler
    samples = rmhmc(rng_key, theta0, eps, L, states)  # Pass in your 'observed' states
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    plt.figure(figsize=(10, 6))
    # Analyze and plot samples (convergence diagnostics, etc.)
    for i,param_name in enumerate(param_names):
        print(f"Parameter: {param_name} : Mean: {jnp.mean(samples[:,i])}, Std: {jnp.std(samples[:,i])}")
        plt.plot(samples[:,i], label=param_name)
    plt.title('Parameter Trace Plots')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/RMHMC_generalized_harmonic.pdf')
if __name__ == "__main__":
    main()