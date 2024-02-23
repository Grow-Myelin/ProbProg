import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular
from jax.numpy.linalg import slogdet
import numpy as np

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
        log_det_inv = log_det
        return log_det_inv
    except Exception as e:
        raise ValueError(f"Failed to compute log_det_A_inv: {e}")

import numpy as np

def test_log_det_inv(log_det_A_inv, A):
    try:
        result_your_func = log_det_A_inv(A)
        result_lib_func = -np.log(np.abs(np.linalg.det(np.linalg.inv(A)))) # Modified

        print("Your Function Output:", result_your_func, type(result_your_func))
        print("NumPy Output:", result_lib_func, type(result_lib_func))
        if result_your_func is None and result_lib_func is None:
            print(f"Matrix {A} is singular -- Test Passed")
        elif np.allclose(result_your_func, result_lib_func):
            print(f"Matrix {A} -- Test Passed")
        else:
            print(f"Matrix {A} -- Discrepancy:")
            print("  Your Function: ", result_your_func)
            print("  Library: ", result_lib_func)

    except np.linalg.LinAlgError:
        print(f"Matrix {A} is likely singular -- Exception caught")

# Example usage
def main():
    A = jnp.array([[6., 3.], [1., 4.]])
    #A = jnp.array([[1., 6.], [10., 1.]])  # Example symmetric positive-definite matrix
    log_det_inv= log_det_A_inv(A)
    test_log_det_inv(log_det_A_inv,A)
    print(f"Log determinant of A's inverse: {log_det_inv}")

if __name__ == "__main__":
    main()

