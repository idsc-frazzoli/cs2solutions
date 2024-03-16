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

        for name, obj in globals().items():
            if obj is A_cl:
                input_name = name
                break
        else: 
            input_name = "the matrix"
        
        print(f"The poles of {input_name} are: {poles}")
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
    assert p_1 is not None, "p_1 must be a float"
    assert p_2 is not None, "p_2 must be a float"
    N = 2  # Number of states

    # Calculate the closed loop characteristic polynomial coefficients a_0 and a_1
    a_0, a_1 = np.poly(np.array([p_1, p_2]))[1:]

    # Create the selection matrix for the Ackermann formula
    select = np.zeros(N)
    select[-1] = 1

    # Calculate the Ackermann formula
    p_cl = A @ A + a_0 * A + a_1 * np.eye(N)
    print("p_cl = ", p_cl)

    # Calculate the control matrix K
    R = sol_control_matrix1(A, B)
    K = np.array([select @ np.linalg.inv(R) @ p_cl])
    print("K = ", K)

    # Compute the state-space matrices of the closed loop system, which takes r as an input
    A_cl = A - B @ K 
    print("A_cl = ", A_cl)
    k_r = -1/(C @ np.linalg.inv(A_cl) @ B)
    k_r = float(k_r[0, 0])
    print("k_r = ", k_r)
    B_cl = B * k_r
    print("A_cl =", A_cl, "\n B_cl =", B_cl)

    return p_cl, K, k_r, A_cl, B_cl

def sol_2x2_acker_estimation(A: np.ndarray, C: np.ndarray, poles: List[float]) -> np.ndarray:
    """
    Calculates the observer gain matrix L for a 2x2 system.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``C`` (np.array): The output matrix.
    - ``poles`` (List[float]): The poles of the observer.

    Returns:
    - ``L```(np.array): The observer gain matrix.
    """
    # Safety
    assert isinstance(A, np.ndarray), "A must be a NumPy array"
    assert isinstance(C, np.ndarray), "C must be a NumPy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == C.shape[1], "A and C must have compatible dimensions"
    assert A.shape[0] == 2, "A must have dimensions larger than 0"
    assert len(poles) == 2, "poles must be a list of length 2"
    N = 2  # Number of states

    # Calculate the observer gain matrix L
    CA = C @ A
    O = np.concatenate((C, CA), axis=0)
    O_inv = np.linalg.inv(O)
    gamma = O_inv @ np.array([[0, 1]]).T

    p_1 = poles[0]*(-1)
    p_2 = poles[1]*(-1)
    ab = p_1 + p_2
    b = p_1*p_2
    p_cl = A @ A + ab*A + b*np.identity(2)

    L = p_cl @ gamma
    return L

def sol_check_pos_def_sym(A: np.ndarray) -> bool:
    """
    Checks if a matrix is positive definite and symmetric.

    Parameters:
    - ``A`` (np.array): The matrix to be checked.

    Returns:
    - bool: True if the matrix is positive definite and symmetric, False otherwise.
    """
    # Safety
    assert isinstance(A, np.ndarray), "A must be a NumPy array"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] > 0, "A must have dimensions larger than 0"

    # Check if the matrix is symmetric
    if not np.allclose(A, A.T):
        return False

    # Check if the matrix is positive definite
    if np.all(np.linalg.eigvals(A) > 0):
        return True
    else:
        return False
    

def test_check_pos_def_sym(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Tests the correctness of the student's solution for the 'sol_check_pos_def_sym' function.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student.
    - ``master_sol`` (callable): The model solution function.
    - ``shouldprint`` (bool): Whether the test results should be printed.

    Returns:
    - bool: The result of the test.
    """
    # Define the matrices to be tested
    A1 = np.array([[6, -2, -3], [-2, 7, 5] , [-3, 5, 6]]) #symmetric and positive definite
    A2 = np.array([[1, -2, -3], [-2, 7, 5] , [-3, 5, 1]]) #symmetric and not positive definite
    A3 = np.array([[3.25, -7.5, -12], [3.375, 17.75, 22] , [-2.0625, -7.125, -8]]) #not symmetric and positive definite
    A4 = np.array([[1, 3, 5], [2, 2, 5] , [-3, 7, 1]]) #not symmetric and not positive definite

    passed_tests = 0

    if shouldprint: print("Student 1: " + str(student_sol(A1)))
    if shouldprint: print("Master 1: " + str(master_sol(A1)))
    if student_sol(A1) == master_sol(A1): passed_tests += 1

    if shouldprint: print("Student 2: " + str(student_sol(A2)))
    if shouldprint: print("Master 2: " + str(master_sol(A2)))
    if student_sol(A2) == master_sol(A2): passed_tests += 1

    if shouldprint: print("Student 3: " + str(student_sol(A3)))
    if shouldprint: print("Master 3: " + str(master_sol(A3)))
    if student_sol(A3) == master_sol(A3): passed_tests += 1

    if shouldprint: print("Student 4: " + str(student_sol(A4)))
    if shouldprint: print("Master 4: " + str(master_sol(A4)))
    if student_sol(A4) == master_sol(A4): passed_tests += 1

    print("Passed tests:", passed_tests, " out of 4")
    return passed_tests == 4


def sol_K_LQR(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: np.ndarray = None) -> np.ndarray:
    """
    Calculate the LQR K gain matrix

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``B`` (np.array): The input matrix.
    - ``Q`` (np.array): The state cost matrix.
    - ``R`` (np.array): The input cost matrix.
    - ``N`` (np.array): The cross-term matrix.

    Returns:
    - ``K_LQR```(np.array): The LQR gain matrix.
    """
    # Safety
    assert A.shape[0] == A.shape[1], "A must be square"
    assert Q.shape[0] == Q.shape[1], "Q must be square"
    assert R.shape[0] == R.shape[1], "R must be square"
    assert B.shape[0] == A.shape[0], "B must have the same number of rows as A"

    if N is None: N = np.zeros((A.shape[0], B.shape[1]))
    if sol_check_pos_def_sym(Q) & sol_check_pos_def_sym(R):
        P = ct.care(A, B, Q, R)[0]
        K_LQR = np.linalg.inv(R) @ np.transpose(N + P @ B)
        return K_LQR
    else:
        return "Q and R must be symmetric and positive definite"