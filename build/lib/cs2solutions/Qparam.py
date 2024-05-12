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

def sol_compute_K_L(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, poles_controller: List[float, float], poles_observer: List[float, float]) -> Tuple[ct. TransferFunction, float, float]:
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

    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert B.shape[0] == A.shape[0], "Matrix B must have the same number of rows as A"
    assert C.shape[1] == A.shape[1], "Matrix C must have the same number of columns as A"
    assert D.shape[0] == C.shape[0] and D.shape[1] == B.shape[1], "Matrix D must have the same number of rows as C and the same number of columns as B"
    assert len(poles_controller) == A.shape[0], "The poles of the controller should have the correct length"
    assert len(poles_observer) == A.shape[0], "The poles of the observer should have the correct length"

    P = ct.ss2tf(A, B, C, D)

    K = ct.acker(A, B, poles_controller)
    L = ct.acker(A.T, C.T, poles_observer).T

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
            'B': np.array([[1, 2], [5, 6], [3, 4]]),
            'C': np.array([[1, -2, 1]]),
            'D': np.array([[1, 3]]),
            'poles_controller': [-1, -2, -3],
            'poles_observer': [-4, -5, -6]
        },{
            'A': np.array([[1, 2], [0, 1]]),
            'B': np.array([[1], [0]]),
            'C': np.array([[1, 0]]),
            'D': np.array([[0]]),
            'poles_controller': [-1, -2],
            'poles_observer': [-3, -4]
        },{
            'A': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            'B': np.array([[1, 0], [0, 1], [0, 0]]),
            'C': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'D': np.array([[0, 0], [0, 0], [0, 0]]),
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
    floats_close = np.allclose([student[1], student[2]], [master[1], master[2]])
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