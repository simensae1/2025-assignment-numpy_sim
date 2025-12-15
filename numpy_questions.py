"""Assignment - using numpy and making a PR.

The goals of this assignment are:
    * Use numpy in practice with two easy exercises.
    * Use automated tools to validate the code (`pytest` and `flake8`)
    * Submit a Pull-Request on github to practice `git`.

The two functions below are skeleton functions. The docstrings explain what
are the inputs, the outputs and the expected error. Fill the function to
complete the assignment. The code should be able to pass the test that we
wrote. To run the tests, use `pytest test_numpy_question.py` at the root of
the repo. It should say that 2 tests ran with success.

We also ask to respect the pep8 convention: https://pep8.org.
This will be enforced with `flake8`. You can check that there is no flake8
errors by calling `flake8` at the root of the repo.
"""
import numpy as np


def max_index(X):
    """Return the index of the maximum in a numpy array.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input array.

    Returns
    -------
    (i, j) : tuple(int)
        The row and columnd index of the maximum.

    Raises
    ------
    ValueError
        If the input is not a numpy array or
        if the shape is not 2D.
    """
    # 1. Validation checks (already correct, but improved style)
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray.")
    # Check for 2D shape (X.ndim == 2)
    if X.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got {X.ndim} dimensions.")

    # 2. Find the index efficiently using NumPy
    # np.argmax(X) finds the index of the maximum value if the array was flattened.
    flat_index = np.argmax(X)
    
    # np.unravel_index converts the flat index back to (row, column) coordinates
    # based on the array's shape.
    row_index, col_index = np.unravel_index(flat_index, X.shape)

    # The docstring asks for (i, j) where i is row and j is column.
    return int(row_index), int(col_index)


def wallis_product(n_terms):
    """Implement the Wallis product to compute an approximation of pi.

    See:
    https://en.wikipedia.org/wiki/Wallis_product

    Parameters
    ----------
    n_terms : int
        Number of steps in the Wallis product. Note that `n_terms=0` will
        consider the product to be `1`.

    Returns
    -------
    pi : float
        The approximation of order `n_terms` of pi using the Wallis product.
    """
    # XXX : The n_terms is an int that corresponds to the number of
    # terms in the product. For example 10000.

    if n_terms == 0:
        return 2.0  # The product part is 1.0, so pi approx is 2 * 1.0
        
    # Initialize the product P. The final approximation for pi is 2 * P.
    product = 1.0
    
    # Iterate from k=1 up to n_terms
    # The Wallis product for pi/2 involves terms indexed by k=1, 2, 3, ... n_terms
    for k in range(1, n_terms + 1):
        # The k-th factor is (4*k^2) / (4*k^2 - 1)
        
        numerator = 4 * k * k
        denominator = 4 * k * k - 1
        
        # Multiply the current product by the k-th factor
        product *= (numerator / denominator)
        
    # The Wallis product approximates pi/2.
    # Therefore, pi is approximated by 2 * product.
    pi_approximation = 2.0 * product
    return pi_approximation
