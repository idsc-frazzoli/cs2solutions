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

def sol_compute_K_L(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, poles_controller: List[float], poles_observer: List[float]) -> Tuple[ct. TransferFunction, float, float]:
    """
    This function computes the transfer function ``P(s)`` of the plant, the controller gain ``K`` and the observer gain ``L``.

    Parameters:
    - ``A`` (numpy.ndarray): The state matrix of the plant.
    - ``B`` (numpy.ndarray): The input matrix of the plant.
    - ``C`` (numpy.ndarray): The output matrix of the plant.
    - ``D`` (numpy.ndarray): The feedforward matrix of the plant.
    - ``poles_controller`` (List[float, float]): The poles of the controller.
    - ``poles_observer`` (List[float, float]): The poles of the observer.

    Returns:
    - ``P`` (ct.TransferFunction): The transfer function of the plant.
    - ``K`` (float): The controller gain.
    - ``L`` (float): The observer gain.
    """
    assert B.shape[1] == 1 and C.shape[0] == 1, "System must be SISO"
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert B.shape[0] == A.shape[0], "Matrix B must have the same number of rows as A"
    assert C.shape[1] == A.shape[1], "Matrix C must have the same number of columns as A"
    assert D.shape[0] == C.shape[0] and D.shape[1] == B.shape[1], "Matrix D must have the same number of rows as C and the same number of columns as B"
    assert len(poles_controller) == A.shape[0], "The poles of the controller should have the correct length"
    assert len(poles_observer) == A.shape[0], "The poles of the observer should have the correct length"

    P = ct.ss2tf(A, B, C, D)

    K = np.array(ct.acker(A, B, poles_controller))
    L = np.array(ct.acker(A.T, C.T, poles_observer).T)

    return P, K, L

def test_compute_K_L(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit tests for the student implementation of the ``compute_K_L`` function.

    Parameters:
    - ``student_sol`` (callable): The implementation of the student's function.
    - ``master_sol`` (callable): The implementation of the master's function.
    - ``shouldprint`` (bool): Whether to print the results of the tests.

    Returns:
    - ``bool``: Whether the tests passed or not.
    """

    test_cases = [
        {
            'A': np.array([[5, 6, 6], [0, 0, 1], [-5, -5, -6]]),
            'B': np.array([[1], [5], [3]]),
            'C': np.array([[1, -2, 1]]),
            'D': np.array([[1]]),
            'poles_controller': [-1, -2, -3],
            'poles_observer': [-4, -5, -6]
        },
        {
            'A': np.array([[1, 2], [0, 1]]),
            'B': np.array([[1], [1]]),
            'C': np.array([[1, 0]]),
            'D': np.array([[0]]),
            'poles_controller': [-1, -2],
            'poles_observer': [-3, -4]
        },
        {
            'A': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            'B': np.array([[1], [2], [1]]),
            'C': np.array([[1, 0, 0]]),
            'D': np.array([[0]]),
            'poles_controller': [-1, -2, -3],
            'poles_observer': [-4, -5, -6]
        }
    ]

    passed_tests = 0
    total_tests = len(test_cases)
    for i, test_case in enumerate(test_cases):
        try:
            student = student_sol(test_case['A'], test_case['B'], test_case['C'], test_case['D'], test_case['poles_controller'], test_case['poles_observer'])
        except Exception as e:
            print(f"Could not run student solution for test case {i}: {e}")
            return False
        master = master_sol(test_case['A'], test_case['B'], test_case['C'], test_case['D'], test_case['poles_controller'], test_case['poles_observer'])
        if check_compute_K_L(student, master, shouldprint):
            passed_tests += 1
    
    print("Passed tests: ", passed_tests, "/", total_tests)
    return passed_tests == total_tests

def check_compute_K_L(student: Tuple[ct. TransferFunction, float, float], master: Tuple[ct. TransferFunction, float, float], shouldprint: bool = True) -> bool:
    """
    Check if the solution between student and master is close.

    Parameters:
    - ``student`` (Tuple[ct. TransferFunction, float, float]): The student's solution.
    - ``master`` (Tuple[ct. TransferFunction, float, float]): The master's solution.

    Returns:
    - ``bool``: Whether the solutions are close or not.
    """
    tf_close = np.allclose(student[0].num, master[0].num) and np.allclose(student[0].den, master[0].den)
    floats_close = np.allclose(student[1], master[1]) and np.allclose(student[2], master[2])
    if tf_close and floats_close:
        return True
    else:
        if shouldprint:
            print("The student's solution is not correct")
            print("Student's transfer function: ", student[0])
            print("Master's transfer function: ", master[0])
            print("Student's K: ", student[1])
            print("Master's K: ", master[1])
            print("Student's L: ", student[2])
            print("Master's L: ", master[2])
        return False
    
def sol_compute_Q_tf_ss_poles() -> Tuple[ct.TransferFunction, ct.StateSpace, np.ndarray]:
    """
    This function computes the transfer function ``Q(s)``, its state-space representation and its poles.

    Returns:
    - ``Q`` (ct.TransferFunction): The transfer function of the system.
    - ``Q_ss`` (ct.StateSpace): The state-space representation of the system.
    - ``poles`` (np.ndarray): The poles of the system.
    """
    Q = ct.TransferFunction([1, 0], [1, 1], name="Q")
    sys_q = ct.tf2ss(Q)
    qpoles = ct.poles(Q)
    print(qpoles)
    return Q, sys_q, qpoles

def sol_compute_ABK_ALC(A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the matrices ``A-BK`` and ``A-LC``.

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system.
    - ``B`` (np.ndarray): The input matrix of the system.
    - ``C`` (np.ndarray): The output matrix of the system.
    - ``K`` (np.ndarray): The controller gain.
    - ``L`` (np.ndarray): The observer gain.

    Returns:
    - ``ABK`` (np.ndarray): The matrix ``A-BK``.
    - ``ALC`` (np.ndarray): The matrix ``A-LC``.
    """
    ABK = A - B @ K
    ALC = A - L @ C
    return ABK, ALC

def sol_compute_total_closed_loop(A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, ABK: np.ndarray, ALC: np.ndarray, sys_q: ct.StateSpace) -> ct.ss:
    """
    This function calculates the total closed-loop system.

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system.
    - ``B`` (np.ndarray): The input matrix of the system.
    - ``C`` (np.ndarray): The output matrix of the system.
    - ``K`` (np.ndarray): The controller gain.
    - ``ABK`` (np.ndarray): The matrix ``A-BK``.
    - ``ALC`` (np.ndarray): The matrix ``A-LC``.
    - ``sys_q`` (ct.StateSpace): The state-space representation of the system.

    Returns:
    - ``sys_cl`` (ct.ss): The total closed-loop system.
    """
    # Define the three additional matrices we need
    BCQ = B @ sys_q.C
    BK = B @ K
    BQC = sys_q.B @ C

    # ----------------------------------------------------------------------------------------------

    # Find the A matrix of the full closed loop system
    #Concatenate the evaluated submatrices to find the first row of the total A matrix
    Acl_1 = np.concatenate((ABK, BCQ, -BK), axis=1)

    # Define the shape of the zeros-matrix at position (2,1)
    zeros_21 = np.zeros((sys_q.A.shape[0], ABK.shape[1]))

    #Concatenate the evaluated submatrices to find the second row of the total A matrix
    Acl_2 = np.concatenate((zeros_21, sys_q.A, BQC), axis=1)

    # Define the shape of the zeros-matrix at position (3, 1)
    zeros_31 = np.zeros((ABK.shape[0], ABK.shape[1]))

    # Define the shape of the zeros-matrix at position (3, 2)
    zeros_32 = np.zeros((A.shape[0], sys_q.A.shape[1]))

    #Concatenate the evaluated submatrices to find the third row of the total A matrix
    Acl_3 = np.concatenate((zeros_31, zeros_32, ALC), axis=1)

    # Concatenate the three evaluated rows to find the total A matrix
    Acl = np.concatenate((Acl_1, Acl_2, Acl_3), axis=0)

    # ----------------------------------------------------------------------------------------------

    # Find the B matrix of the full closed loop system
    # Find the components of the first row of the total B matrix
    B_11 = B
    B_12 = np.zeros((ABK.shape[0], 1))

    # Find the first row of the total B matrix
    B_1 = np.concatenate((B_11, B_12), axis=1)

    # Find the components of the first row of the total B matrix
    B_21 = np.zeros((ABK.shape[0] + sys_q.A.shape[0], 1))
    B_22 = np.ones((ABK.shape[0] + sys_q.A.shape[0], 1))

    # Find the second row of the total B matrix
    B_2 = np.concatenate((B_21, B_22), axis=1)

    # Concatenate the three evaluated rows to find the total B matrix
    Bcl = np.concatenate((B_1, B_2), axis=0)

    # ----------------------------------------------------------------------------------------------

    # Find the C matrix of the full closed loop system
    C_1 = np.zeros((1, Acl.shape[0]))
    C_1[0][0] = 1
    C_2 = np.concatenate((np.zeros((C.shape)), sys_q.C, np.zeros((C.shape))), axis=1)

    # Concatenate the two evaluated rows to find the total C matrix
    Ccl = np.concatenate((C_1, C_2), axis=0)

    # ----------------------------------------------------------------------------------------------

    # Find the D matrix of the full closed loop system
    Dcl = np.zeros((2, 2))

    # ----------------------------------------------------------------------------------------------s

    # Define the full closed loop system
    syscl = ct.ss(Acl, Bcl, Ccl, Dcl)

    return syscl, Acl, Bcl, Ccl, Dcl

def sol_makeweight_1(M: float, w1: float) -> ct.TransferFunction:
    """
    Solution to making a weight transfer function with form M/(M/omega*s + 1).

    Parameters:
    - ``M`` (float): The magnitude of the weight.
    - ``w1`` (float): The frequency of the weight.

    Returns:
    - ``W1`` (ct.TransferFunction): The weight transfer function.
    """
    return ct.TransferFunction([M], [M/w1, 1], name='W_1')

def sol_makeweight_3(M: float, w3: float) -> ct.TransferFunction:
    """
    Solution to making a weight transfer function with form Ms/(s+M*omega).

    Parameters:
    - ``M`` (float): The magnitude of the weight.
    - ``w3`` (float): The frequency of the weight.

    Returns:
    - ``W3`` (ct.TransferFunction): The weight transfer function.
    """
    return ct.TransferFunction([M, 0], [1, M*w3], name='W_3')

def test_same_tf(student: ct.TransferFunction, master: ct.TransferFunction, shouldprint: bool = True) -> bool:
    """
    Simple function to check if two transfer functions are equivalent.

    Parameters:
    - ``student`` (ct.TransferFunction): The student's transfer function.
    - ``master`` (ct.TransferFunction): The master's transfer function.

    Returns:
    - ``bool``: Whether the transfer functions are equivalent.
    """
    if np.allclose(student.num, master.num) and np.allclose(student.den, master.den):
        return True
    else:
        if shouldprint:
            print("The student's solution is not correct")
            print("Student's transfer function: ", student)
            print("Master's transfer function: ", master)
        return False

