from rmhmc_test import riemann_metric
import jax
from typing import Tuple, Callable
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, cholesky,solve_triangular
from jax.scipy.special import logsumexp, logit, expit as sigmoid
from jax.scipy.stats import norm
from matrixOperations import log_det_A_inv


def test_riemann_metric_2d_basic():
    # 2D parameter space
    x = jnp.array([0.5, -0.3])
    y = jnp.array([1, -1])
    K = jnp.array([[1.0, 0.2], [0.2, 1.0]])
    LK = cholesky(K, lower=True)
    log_det_K = jnp.log(jnp.linalg.det(K))
    mu = jnp.array([0.0, 0.0])
    Sigma = K  # Identical to K for simplicity
    LSigma = LK  # Identical to LK
    log_det_Sigma = log_det_K  # Identical to log_det_K
    beta = 0.5
    
    L,dLdx, G_inv, log_det_G, dg = riemann_metric(x, y, K, LK, log_det_K, mu, Sigma, LSigma, log_det_Sigma, beta)
    print(dLdx)
    # Assertions: Check shapes
    assert G_inv.shape == (2, 2), "G_inv shape mismatch"
    assert isinstance(log_det_G, float) or log_det_G.ndim == 0, "log_det_G should be scalar"
    assert dg.shape == x.shape, "dg shape mismatch"

def test_riemann_metric_identity_kernel():
    x = jnp.array([1.0, -1.0, 0.5])
    y = jnp.array([1, 1, -1])
    K = jnp.eye(3)  # Identity matrix
    LK = cholesky(K, lower=True)
    log_det_K = 0.0  # Log determinant of identity matrix is 0
    mu = jnp.zeros(3)
    Sigma = K  # Same as K
    LSigma = LK  # Same as LK
    log_det_Sigma = 0.0  # Same as log_det_K
    beta = 1.0
    
    L, dLdx, G_inv, log_det_G, dg = riemann_metric(x, y, K, LK, log_det_K, mu, Sigma, LSigma, log_det_Sigma, beta)
    print(G_inv)
    print(K)
    print(log_det_G)
    # Assertions: G_inv should be close to K (identity) and log_det_G should be 0
#    assert jnp.allclose(G_inv, K), "G_inv mismatch for identity kernel"
    assert jnp.isclose(log_det_G, 0.0), "log_det_G mismatch for identity kernel"


def test_riemann_metric_non_positive_definite_kernel():
    x = jnp.array([0.1, -0.2])
    y = jnp.array([1, -1])
    K = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive-definite
    try:
        LK = cholesky(K, lower=True)
        log_det_K = jnp.log(jnp.linalg.det(K))
        mu = jnp.array([0.0, 0.0])
        Sigma = K
        LSigma = cholesky(Sigma, lower=True)
        log_det_Sigma = jnp.log(jnp.linalg.det(Sigma))
        beta = 1.0
        
        riemann_metric(x, y, K, LK, log_det_K, mu, Sigma, LSigma, log_det_Sigma, beta)
        assert False, "Expected an error due to non-positive-definite kernel matrix"
    except jax.lib.xla_extension.StatusNotOk as e:
        assert "cholesky" in str(e), "Expected Cholesky decomposition failure"


def main():
    test_riemann_metric_2d_basic()
    test_riemann_metric_identity_kernel()
    test_riemann_metric_non_positive_definite_kernel()

if __name__ == "__main__":
    main()