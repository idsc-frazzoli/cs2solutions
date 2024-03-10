from typing import List, Tuple

try:
    from typing import Optional, List, Tuple
    import numpy as np
    import matplotlib.pyplot as plt
    import control as ct
    from scipy import signal as sig
    from scipy.signal import butter, lfilter
    import unittest as unit
except ImportError as e:
    print(f"Error: {e}")
    print(f"Please install the required packages using the command '!pip install control numpy matplotlib scipy'")


def sol_print_poles(*args: np.ndarray) -> list:
    """
    Prints the poles of the modified A_cl=A-BK matrix.

    Parameters:
    - ``args``: Variable-length list of np.ndarray respresenting state matrices of various systems

    Returns:
    - list: List of arrays containing the poles of each system
    """
    assert args, "At least one np.ndarray argument is required"
    
    poles_list = []
    for A_cl in args:
        # Safety
        assert isinstance(A_cl, np.ndarray), "Input matrix must be a NumPy array"
        assert len(A_cl.shape) == 2, "Input matrix must be a 2D array"
        assert A_cl.shape[0] == A_cl.shape[1], "Input matrix must be square"
        assert A_cl.shape[0] > 0, "Input matrix dimensions must be larger than 0"

        # Calculate eigenvalues
        poles = np.linalg.eigvals(A_cl)
        print(f"The poles of the modified system are: {poles}")
        poles_list.append(poles)
    return poles_list


def sol_control_matrix1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Create the controllability matrix R of a system given the state-space representation.
    Variation 1.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``B`` (np.array): The input matrix.

    Returns:
    - ``R```(np.array): Controllability matrix.
    """
    # Safety
    assert isinstance(A, np.ndarray), "A must be a NumPy array"
    assert isinstance(B, np.ndarray), "B must be a NumPy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "A and B must have compatible dimensions"
    assert A.shape[0] > 0, "A must have dimensions larger than 0"
    assert B.shape[1] > 0, "B must have dimensions larger than 0"

    n = A.shape[0]  # Get the number of rows in matrix A.
    R = B  # Initialize the controllability matrix R with matrix B, since the first column of R is always B.
    
    for i in range(1, n):  # Iterate from 1 to n-1.
        # Calculate the reachability matrix for each power of A and concatenate it horizontally to R.
        R = np.hstack((R, np.linalg.matrix_power(A, i) @ B))
    
    return R


def sol_control_matrix2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Create the controllability matrix R of a system given the state-space representation.
    Variation 2.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``B`` (np.array): The input matrix.

    Returns:
    - ``R```(np.array): Controllability matrix.
    """
    # Safety
    assert isinstance(A, np.ndarray), "A must be a NumPy array"
    assert isinstance(B, np.ndarray), "B must be a NumPy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "A and B must have compatible dimensions"
    assert A.shape[0] > 0, "A must have dimensions larger than 0"
    assert B.shape[1] > 0, "B must have dimensions larger than 0"
    
    n = A.shape[0]  # number of states
    m = B.shape[1]  # number of inputs
    print("n =", n, ", m =", m)

    # Initialize R with the right dimensions
    R = np.zeros((n, n*m))  

    # Fill R using the reachability matrix formula
    R[:, :m] = B
    for i in range(1, n):
        R[:, i*m:(i+1)*m] = np.dot(A, R[:, (i-1)*m:i*m])

    return R


def sol_2x2_ackermann(A: np.ndarray, B: np.ndarray, C:np.ndarray, p_1: float, p_2: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Calculates and outputs all steps of the Ackermann formula in one go. Only works for 2x2 matrices.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``B`` (np.array): The input matrix.
    - ``C`` (np.array): The output matrix.
    - ``p_1`` (float): The first pole.
    - ``p_2`` (float): The second pole.

    Returns:
    - ``p_cl`` (np.array): Result of the Ackermann formula multiplication
    - ``K`` (np.array): The control matrix [0 ... 0 1] @ R^{-1} @ p_cl(A)
    - ``k_r`` (float): The gain of the controller.
    - ``A_cl`` (np.array): The closed-loop state matrix.
    - ``B_cl`` (np.array): The closed-loop input matrix.
    """
    # Safety
    assert isinstance(A, np.ndarray), "A must be a NumPy array"
    assert isinstance(B, np.ndarray), "B must be a NumPy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "A and B must have compatible dimensions"
    assert A.shape[0] == 2, "A must have dimensions larger than 0"
    assert B.shape[1] == 2, "B must have dimensions larger than 0"
    N = 2  # Number of states

    # Calculate the closed loop characteristic polynomial coefficients a_0 and a_1
    a_0, a_1 = np.poly(np.array([p_1, p_2]))[1:]

    # Create the selection matrix for the Ackermann formula
    select = np.zeros(N)
    select[-1] = 1

    # Calculate the Ackermann formula
    p_cl = A @ A + a_0 * A + a_1 * np.eye(N)
    print("p_cl = ", p_cl)
    K = np.array([select @ np.linalg.inv(R) @ p_cl])
    print("K = ", K)

    # Compute the state-space matrices of the closed loop system, which takes r as an input
    A_cl = A - B @ K 
    print("A_cl = ", A_cl)
    k_r = -1/(C @ np.linalg.inv(A_cl) @ B)
    print("k_r = ", k_r)
    B_cl = B * k_r
    print("A_cl =", A_cl, "\n B_cl =", B_cl)

    return p_cl, K, k_r, A_cl, B_cl