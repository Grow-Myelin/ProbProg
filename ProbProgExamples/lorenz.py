import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey
from jax import jit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

matplotlib.use("Agg")  # noqa: E402

@jit
def dz_dt(z, t, theta):
    """
    Computes the derivative of the Lorenz system at a point.

    Parameters:
    - z: jnp.ndarray, current state of the system [x, y, z].
    - t: float, current time (unused as Lorenz equations are autonomous).
    - theta: jnp.ndarray, parameters of the system [sigma, rho, beta].

    Returns:
    - jnp.ndarray, derivative [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = z
    sigma, rho, beta = theta
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return jnp.array([dx_dt, dy_dt, dz_dt])


def generate_synthetic_data(theta, z_init, N):
    """
    Generates synthetic data using the Lorenz model.

    Parameters:
    - theta: jnp.ndarray, parameters for the Lorenz model.
    - z_init: jnp.ndarray, initial state of the system.
    - N: int, number of data points to generate.

    Returns:
    - jnp.ndarray, generated data points.
    """
    ts = jnp.linspace(0.0, N-1, N)
    z_init = jnp.array(z_init, dtype=jnp.float32)
    theta = jnp.array(theta, dtype=jnp.float32)

    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    return z


def model(N, y=None):
    """
    Bayesian model for inference using the Lorenz system.

    Parameters:
    - N: int, number of data points.
    - y: jnp.ndarray, observed data points.
    """
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([3]))
    ts = jnp.linspace(0.0, N-1, N)
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([10.0, 28.0, 8.0 / 3.0]),
            scale=jnp.array([0.5, 0.5, 0.5]),
        ),
    )
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)

    sigma = numpyro.sample("sigma", dist.Exponential(1).expand([3]))
    numpyro.sample("y", dist.Normal(z, sigma), obs=y)


def main(args):
    """
    Main function to perform Bayesian inference and plot results.

    Parameters:
    - args: argparse.Namespace, command line arguments.
    """
    # Parameters for synthetic data
    true_theta = jnp.array([10.0, 28.0, 8.0 / 3.0])
    z_init = jnp.array([1.0, 1.0, 1.0])
    N = 30

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(true_theta, z_init, N)

    # Bayesian inference
    mcmc = MCMC(
        NUTS(model, dense_mass=True),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(PRNGKey(1), N=N, y=synthetic_data)
    mcmc.print_summary()

    # Prediction
    pred = Predictive(model, mcmc.get_samples())(PRNGKey(2), N)["y"]
    mu = jnp.mean(pred, 0)
    pi = jnp.percentile(pred, jnp.array([10, 90]), 0)

    # Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(synthetic_data[:, 0], synthetic_data[:, 1], synthetic_data[:, 2], "r-", label="synthetic data")
    ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], "b--", label="predicted")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.title("Lorenz Attractor: Synthetic Data and Predicted Trajectory")
    plt.legend()

    plt.savefig("plots/lorenz_plot.pdf")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.13.2")
    parser = argparse.ArgumentParser(description="Lorenz Model")
    parser.add_argument("-n", "--num-samples", nargs="?", default=500, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=100, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
