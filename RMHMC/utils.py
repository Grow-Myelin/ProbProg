import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular
from jax.numpy.linalg import slogdet

@jit
def log_det_A_inv(A: jnp.ndarray) -> float:
    """
    Computes the log determinant of the inverse of a matrix A.
    
    Parameters:
    - A: jnp.ndarray. A symmetric positive-definite matrix.
    
    Returns:
    - log_det_inv: float. The log determinant of the inverse of A.
    
    Note:
    - The function assumes A is symmetric and positive-definite.
    - Utilizes Cholesky decomposition for numerical stability and efficiency.
    """
    try:
        # Perform Cholesky decomposition of A
        L = cholesky(A, lower=True)
        
        # Compute the log determinant of A using the Cholesky factor
        # log(det(A)) = 2 * sum(log(diag(L)))
        log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
        
        # The log determinant of the inverse is the negative of log_det(A)
        log_det_inv = -log_det
        return log_det_inv
    except Exception as e:
        raise ValueError(f"Failed to compute log_det_A_inv: {e}")

# Example usage
def main():
    A = jnp.array([[4., 1.], [1., 3.]])  # Example symmetric positive-definite matrix
    log_det_inv = log_det_A_inv(A)
    print(f"Log determinant of A's inverse: {log_det_inv}")

if __name__ == "__main__":
    main()

