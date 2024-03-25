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


def sol_kalman_filter(A: np.ndarray, B: np.ndarray, C: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the L observer gain matrix according to the Kalman Filter definition.

    Parameters:
    - ``A`` (np.ndarray): State matrix of the system.
    - ``B`` (np.ndarray): Input matrix of the system.
    - ``C`` (np.ndarray): Output matrix of the system.
    - ``U`` (np.ndarray): Noisy input (optional).

    Returns:
    - ``L`` (np.ndarray): Kalman estimator gain matrix.
    """
    # Assert that A, B, C are numpy arrays
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert isinstance(C, np.ndarray), "C should be a numpy array"

    # Assert that A, B, C have correct dimensions
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == A.shape[0], "The number of rows in B should be equal to the number of rows in A"
    assert C.shape[1] == A.shape[0], "The number of columns in C should be equal to the number of rows in A"

    #Process and sensor noise covariance matrices
    QN = np.cov(U, rowvar=True)[1, 1]
    RN = QN*10

    # If non gaussian-noise
    if QN == 0:
        QN = 1
        RN = 10

    # Kalman estimator gain
    L, P, E = ct.lqe(A, B, C, QN, RN)
    return L

def test_kalman_filter(student_sol: callable, master_sol: callable, U: np.ndarray, shouldprint: bool = True) -> bool:
    """
    Test the student's Kalman Filter implementation.

    Parameters:
    - ``student`` (callable): The student's Kalman Filter function.
    - ``master`` (callable): The master's Kalman Filter function.
    - ``U`` (np.ndarray): Noisy input.
    - ``shouldprint`` (bool): Whether to print the test results.

    Returns:
    - ``bool``: The test result.
    """
    # Define the system matrices
    A1 = np.array([[0, 1], [-1, -1]])
    B1 = np.array([[0], [1]])
    C1 = np.array([[1, 0]])

    passed_tests = 0
    if student_sol(A1, B1, C1, U) is None:
        print("Student solution is not implemented yet")
        return False

    if shouldprint: print("Student 1: " + str(student_sol(A1, B1, C1, U)))
    if shouldprint: print("Master 1: " + str(master_sol(A1, B1, C1, U)))
    if np.allclose(student_sol(A1, B1, C1, U), master_sol(A1, B1, C1, U)): passed_tests += 1

    A2 = np.array([[0, 1], [-1, -1]])
    B2 = np.array([[0], [1]])
    C2 = np.array([[0, 1]])

    if shouldprint: print("Student 2: " + str(student_sol(A2, B2, C2, U)))
    if shouldprint: print("Master 2: " + str(master_sol(A2, B2, C2, U)))
    if np.allclose(student_sol(A2, B2, C2, U), master_sol(A2, B2, C2, U)): passed_tests += 1

    A3 = np.array([[0, 0.1], [0, 0]])
    B3 = np.array([[0], [1]])
    C3 = np.array([[1, 0]])

    if shouldprint: print("Student 3: " + str(student_sol(A3, B3, C3, U)))
    if shouldprint: print("Master 3: " + str(master_sol(A3, B3, C3, U)))
    if np.allclose(student_sol(A3, B3, C3, U), master_sol(A3, B3, C3, U)): passed_tests += 1

    print("Passed tests:", passed_tests, " out of 3")
    return passed_tests == 3


def kalman_perf_written_ans() -> None:
    """
    The Kalman filter performs well with gaussian noise. Process and sensor noise covariance are difficult to estimate in reality. If the mean and the standard deviation of the noise are known, the estimator behaves better.
    """
    return None


def sol_LQR_K(A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Calculate the LQR controller gain matrix.

    Parameters:
    - ``A`` (np.ndarray): State matrix of the system.
    - ``B`` (np.ndarray): Input matrix of the system.
    - ``C`` (np.ndarray): Output matrix of the system.
    - ``Q`` (np.ndarray): State cost matrix.
    - ``R`` (np.ndarray): Input cost matrix.

    Returns:
    - ``K`` (np.ndarray): LQR controller gain matrix.
    """
    # Assert that A, B, C, Q, R are numpy arrays
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert isinstance(C, np.ndarray), "C should be a numpy array"
    assert isinstance(Q, np.ndarray), "Q should be a numpy array"
    assert isinstance(R, np.ndarray), "R should be a numpy array"

    # Assert that A, B, C, Q, R have correct dimensions
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == A.shape[0], "The number of rows in B should be equal to the number of rows in A"
    assert C.shape[1] == A.shape[0], "The number of columns in C should be equal to the number of rows in A"
    assert Q.shape[0] == Q.shape[1] == A.shape[0], "Q should be a square matrix with the same dimensions as A"
    assert R.shape[0] == B.shape[1], "R should be a square matrix with the same number of rows as the number of columns in B"

    sys = ct.ss(A, B, C, 0)
    K, S, E = ct.lqr(sys, Q, R)
    return K


def test_LQR_K(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Test the student's LQR controller gain implementation.

    Parameters:
    - ``student`` (callable): The student's LQR controller gain function.
    - ``master`` (callable): The master's LQR controller gain function.
    - ``shouldprint`` (bool): Whether to print the test results.

    Returns:
    - ``bool``: The test result.
    """
    # Define the system matrices
    A1 = np.array([[0, 1], [-1, -1]])
    B1 = np.array([[0], [1]])
    C1 = np.array([[1, 0]])
    # Straight path
    Q = 1 * np.eye(2,dtype=int)
    R = np.array([10])
    # Easy/Hard path
    #Q = np.array([[1000,0],[0,10]])
    #R = np.array([1])

    passed_tests = 0
    if student_sol(A1, B1, C1, Q, R) is None:
        print("Student solution is not implemented yet")
        return False

    if shouldprint: print("Student 1: " + str(student_sol(A1, B1, C1, Q, R)))
    if shouldprint: print("Master 1: " + str(master_sol(A1, B1, C1, Q, R)))
    if np.allclose(student_sol(A1, B1, C1, Q, R), master_sol(A1, B1, C1, Q, R)): passed_tests += 1

    A2 = np.array([[0, 1], [-1, -1]])
    B2 = np.array([[0], [1]])
    C2 = np.array([[0, 1]])

    if shouldprint: print("Student 2: " + str(student_sol(A2, B2, C2, Q, R)))
    if shouldprint: print("Master 2: " + str(master_sol(A2, B2, C2, Q, R)))
    if np.allclose(student_sol(A2, B2, C2, Q, R), master_sol(A2, B2, C2, Q, R)): passed_tests += 1

    A3 = np.array([[0, 0.1], [0, 0]])
    B3 = np.array([[0], [1]])
    C3 = np.array([[1, 0]])

    if shouldprint: print("Student 3: " + str(student_sol(A3, B3, C3, Q, R)))
    if shouldprint: print("Master 3: " + str(master_sol(A3, B3, C3, Q, R)))
    if np.allclose(student_sol(A3, B3, C3, Q, R), master_sol(A3, B3, C3, Q, R)): passed_tests += 1

    print("Passed tests:", passed_tests, " out of 3")
    return passed_tests == 3
    

def sol_stability_analysis(A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, L: np.ndarray, shouldprint: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the eigenvalues of the controlled and observed systems.

    Parameters:
    - ``A`` (np.ndarray): State matrix of the system.
    - ``B`` (np.ndarray): Input matrix of the system.
    - ``C`` (np.ndarray): Output matrix of the system.
    - ``K`` (np.ndarray): LQR controller gain matrix.
    - ``L`` (np.ndarray): Kalman estimator gain matrix.
    - ``shouldprint`` (bool): Whether to print the eigenvalues.

    Returns:
    - ``con_eig`` (np.ndarray): Controller eigenvalues.
    - ``obs_eig`` (np.ndarray): Observer eigenvalues.
    """
    # Assert that A, B, C, K, L are numpy arrays
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert isinstance(C, np.ndarray), "C should be a numpy array"
    assert isinstance(K, np.ndarray), "K should be a numpy array"
    assert isinstance(L, np.ndarray), "L should be a numpy array"

    # Assert that A, B, C, K, L have correct dimensions
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == A.shape[0], "The number of rows in B should be equal to the number of rows in A"
    assert C.shape[1] == A.shape[0], "The number of columns in C should be equal to the number of rows in A"
    assert K.shape[0] == B.shape[1], "The number of rows in K should be equal to the number of columns in B"
    assert L.shape[1] == C.shape[0], "The number of columns in L should be equal to the number of rows in C"


    con_eig = np.linalg.eigvals(A - B @ K)
    obs_eig = np.linalg.eigvals(A - L @ C)

    if shouldprint: print("Controller eigenvalues:", con_eig, "Observer eigenvalues:", obs_eig)
    return con_eig, obs_eig


def sol_close_loop(A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, U: np.ndarray) -> ct.StateSpace:
    """
    Calculate the state-space representation of the closed-loop system.

    Parameters:
    - ``A`` (np.ndarray): State matrix of the system.
    - ``B`` (np.ndarray): Input matrix of the system.
    - ``C`` (np.ndarray): Output matrix of the system.
    - ``Q`` (np.ndarray): State cost matrix.
    - ``R`` (np.ndarray): Input cost matrix.
    - ``U`` (np.ndarray): Noisy input.

    Returns:
    - ``sys_cl`` (StateSpace): Closed-loop system.
    """
    # Assert that A, B, C, Q, R are numpy arrays
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert isinstance(C, np.ndarray), "C should be a numpy array"
    assert isinstance(Q, np.ndarray), "Q should be a numpy array"
    assert isinstance(R, np.ndarray), "R should be a numpy array"

    # Assert that A, B, C, Q, R have correct dimensions
    assert A.shape[0] == A.shape[1], "A should be a square matrix"
    assert B.shape[0] == A.shape[0], "The number of rows in B should be equal to the number of rows in A"
    assert C.shape[1] == A.shape[0], "The number of columns in C should be equal to the number of rows in A"
    assert Q.shape[0] == Q.shape[1] == A.shape[0], "Q should be a square matrix with the same dimensions as A"
    assert R.shape[0] == B.shape[1], "R should be a square matrix with the same number of rows as the number of columns in B"

    # Calculate K and L using LQR and Kalman Filter respectively
    K = sol_LQR_K(A, B, C, Q, R)
    L = sol_kalman_filter(A, B, C, U)

    LC = L @ C
    BK = B @ K
    KS = -np.linalg.inv(C @ np.linalg.inv(A-BK) @ B)

    # Create the SISO closed-loop system
    A_s = np.concatenate((A-BK, BK), axis=1)
    A_i = np.concatenate((np.zeros_like(LC), A-LC), axis=1)
    A_cl = np.concatenate((A_s, A_i), axis=0)
    B_cl = np.block([[B @ KS], [np.zeros_like(B)]])
    C_cl = np.block([[C, np.zeros_like(C)]])
    D_cl = 0

    closed_loop_sys_sol = ct.ss(A_cl, B_cl, C_cl, D_cl)
    return closed_loop_sys_sol


def test_close_loop(student_sol: callable, master_sol: callable, U: np.ndarray, shouldprint: bool = True) -> bool:
    """
    Test the student's closed-loop system implementation.

    Parameters:
    - ``student`` (callable): The student's closed-loop system function.
    - ``master`` (callable): The master's closed-loop system function.
    - ``U`` (np.ndarray): Noisy input.
    - ``shouldprint`` (bool): Whether to print the test results.

    Returns:
    - ``bool``: The test result.
    """
    # Define the system matrices
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    # Straight path
    Q = 1 * np.eye(2,dtype=int)
    R = np.array([10])

    passed_tests = 0
    if student_sol(A, B, C, Q, R, U) is None:
        print("Student solution is not implemented yet")
        return False
    
    student_solution = student_sol(A, B, C, Q, R, U)
    master_solution = master_sol(A, B, C, Q, R, U)

    if shouldprint: print("Student A_cl: " + str(student_solution.A))
    if shouldprint: print("Master A_cl: " + str(master_solution.A))
    if np.allclose(student_solution.A, master_solution.A): passed_tests += 1

    if shouldprint: print("Student B_cl: " + str(student_solution.B))
    if shouldprint: print("Master B_cl: " + str(master_solution.B))
    if np.allclose(student_solution.B, master_solution.B): passed_tests += 1

    if shouldprint: print("Student C_cl: " + str(student_solution.C))
    if shouldprint: print("Master C_cl: " + str(master_solution.C))
    if np.allclose(student_solution.C, master_solution.C): passed_tests += 1

    if shouldprint: print("Student D_cl: " + str(student_solution.D))
    if shouldprint: print("Master D_cl: " + str(master_solution.D))
    if np.allclose(student_solution.D, master_solution.D): passed_tests += 1

    print("Passed tests:", passed_tests, " out of 4")
    return passed_tests == 4