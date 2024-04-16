from typing import Tuple

try:
    from typing import Optional, List, Tuple
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    import control as ct
    import signal
    from scipy import signal as signal
    from scipy.signal import butter, lfilter
    import unittest as unit
except ImportError as e:
    print(f"Error: {e}")
    print(f"Please install the required packages using the command '!pip install control numpy matplotlib scipy'")


def sol_sys_matrices():
    """
    Produce the system matrices for the CS2Bot system.
  
    Parameters:
    - None

    Returns:
    - () -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The system matrices A, B, C, and D.
    """
    A = np.array([[0, 3], [0, 0]], dtype=int)
    B = np.array([[0, 0], [0, 1]], dtype=int)
    C = np.array([[1, 0], [0, 1]], dtype=int)
    D = np.array([[0, 0], [0, 0]], dtype=int)

    return A, B, C, D


def test_sys_matrices(student_sol: callable, master_sol: callable) -> bool:
    """
    Check if the student solution is correct for sys_matrices().

    Parameters:
    - ``student_sol`` (function): A function that returns the student's solution.
    - ``master_sol`` (function): A function that returns the master solution.
    - ``shouldprint`` (bool): A boolean indicating whether to print the matrices. Default is True.

    Returns:
    - bool: A boolean indicating if the student solution is correct.
    """
    try:
        student = student_sol()
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol()
    assert np.allclose(student[0], master[0]), "A matrix is incorrect."
    assert np.allclose(student[1], master[1]), "B matrix is incorrect."
    assert np.allclose(student[2], master[2]), "C matrix is incorrect."
    assert np.allclose(student[3], master[3]), "D matrix is incorrect."

    print("Student solution is correct.")
    return True


def sol_poles(A_: np.ndarray, B_: np.ndarray, C_: np.array, D_=np.array) -> int:
    """
    Determine the poles the system.

    Parameters:
    - ``A_`` (np.ndarray): The A matrix of the system.
    - ``B_`` (np.ndarray): The B matrix of the system.
    - ``C_`` (np.ndarray): The C matrix of the system.
    - ``D_`` (np.ndarray): The D matrix of the system.

    Returns:
    - int: The number of inputs for the system.
    """
    return ct.poles(ct.StateSpace(A_, B_, C_, D_))


def test_poles(student_sol: callable, master_sol: callable) -> bool:
    """
    Unit tests for checking the implementation of calc_symb_sol(). This function should symbolically calculate the MIMO transfer function given A, B, C, and D matrices.
    """

    passed_tests = 0
    A1 = np.array([[5, 6, 6], [0, 0, 1], [-5, -5, -6]])
    B1 = np.array([[1, 2], [5, 6], [3, 4]])
    C1 = np.array([[1, -2, 1]])
    D1 = np.array([[1, 3]])
    try:
        student = student_sol(A1, B1, C1, D1)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A1, B1, C1, D1)
    if not np.array_equal(student, master, equal_nan=True):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    A2 = np.array([[0, 3], [0, 0]], dtype=int)
    B2 = np.array([[0, 0], [0, 1]], dtype=int)
    C2 = np.array([[1, 0], [0, 1]], dtype=int)
    D2 = np.array([[0, 0], [0, 0]], dtype=int)
    try:
        student = student_sol(A2, B2, C2, D2)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A2, B2, C2, D2)
    if not np.array_equal(student, master, equal_nan=True):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    A3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    B3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    C3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    D3 = np.array([[0, 0], [0, 0], [0, 0]], dtype=int)
    try:
        student = student_sol(A3, B3, C3, D3)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A3, B3, C3, D3)
    if not np.array_equal(student, master, equal_nan=True):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    print("Passed tests:", passed_tests, " out of 3")
    return passed_tests == 3


def sol_num_inputs(B_: np.ndarray) -> int:
    """
    Determine the number of inputs for the system.

    Parameters:
    - ``A_`` (np.ndarray): The A matrix of the system.
    - ``B_`` (np.ndarray): The B matrix of the system.
    - ``C_`` (np.ndarray): The C matrix of the system.
    - ``D_`` (np.ndarray): The D matrix of the system.

    Returns:
    - int: The number of inputs for the system.
    """
    return B_.shape[1]


def sol_num_outputs(C_: np.ndarray) -> int:
    """
    Determine the number of outputs for the system.

    Parameters:
    - ``A_`` (np.ndarray): The A matrix of the system.
    - ``B_`` (np.ndarray): The B matrix of the system.
    - ``C_`` (np.ndarray): The C matrix of the system.
    - ``D_`` (np.ndarray): The D matrix of the system.

    Returns:
    - int: The number of outputs for the system.
    """
    return C_.shape[0]


def test_num_inputs_outputs(student_num_inputs: callable, student_num_outputs: callable, master_num_inputs: callable,
                            master_num_outputs: callable) -> bool:
    """
    Check if the student solution is correct for num_inputs() and num_outputs().

    Parameters:
    - ``student_num_inputs`` (function): A function that returns the student's solution for num_inputs().
    - ``student_num_outputs`` (function): A function that returns the student's solution for num_outputs().
    - ``master_num_inputs`` (function): A function that returns the master solution for num_inputs().
    - ``master_num_outputs`` (function): A function that returns the master solution for num_outputs().
    - ``shouldprint`` (bool): A boolean indicating whether to print the results. Default is True.

    Returns:
    - bool: A boolean indicating if the student solution is correct.
    """
    B_ = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    C_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    try:
        student_inputs = student_num_inputs(B_)
        student_outputs = student_num_outputs(C_)
    except Exception as e:
        print(f"Error: {e}")
        return False

    master_inputs = master_num_inputs(B_)
    master_outputs = master_num_outputs(C_)
    assert student_inputs == master_inputs, "The number of inputs is incorrect: {student_inputs} != {master_inputs}"
    assert student_outputs == master_outputs, "The number of outputs is incorrect: {student_outputs} != {master_outputs}"

    print("Student solution is correct.")
    return True


def sol_calc_tf_symb(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> sp.Matrix:
    """
    Calculate the transfer function of the system using symbolic mathematics.

    Parameters:
    - ``A`` (np.ndarray): The A matrix of the system.
    - ``B`` (np.ndarray): The B matrix of the system.
    - ``C`` (np.ndarray): The C matrix of the system.
    - ``D`` (np.ndarray): The D matrix of the system.

    Returns:
    - sp.Matrix: The transfer function of the system.
    """
    # Check dimensions of A, B, C, D
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "The number of rows in A and B must be the same"
    assert A.shape[1] == C.shape[1], "The number of columns in A and C must be the same"
    assert B.shape[1] == D.shape[1] and C.shape[0] == D.shape[0], "D must be of dimensions (C.shape[0], B.shape[1])"

    # define identity matrix of appropriate size
    s = sp.symbols('s')
    I = np.eye(A.shape[0])

    # calculate the inverse of (s*I - A)
    term = sp.Matrix(s * I - A)
    inv_term = term.inv()

    # calculate P_s
    transferfunction = np.dot(C, np.dot(inv_term, B)) + D

    # simplify the answer so that there are no double fractions
    transferfunction = sp.simplify(transferfunction)

    return transferfunction


def test_calc_tf_symb(student_sol: callable, master_sol: callable) -> bool:
    """
    Unit tests for checking the implementation of calc_symb_sol(). This function should symbolically calculate the MIMO transfer function given A, B, C, and D matrices.
    """

    passed_tests = 0
    A1 = np.array([[5, 6, 6], [0, 0, 1], [-5, -5, -6]])
    B1 = np.array([[1, 2], [5, 6], [3, 4]])
    C1 = np.array([[1, -2, 1]])
    D1 = np.array([[1, 3]])
    try:
        student = student_sol(A1, B1, C1, D1)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A1, B1, C1, D1)
    if (student != master):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    A2 = np.array([[0, 3], [0, 0]], dtype=int)
    B2 = np.array([[0, 0], [0, 1]], dtype=int)
    C2 = np.array([[1, 0], [0, 1]], dtype=int)
    D2 = np.array([[0, 0], [0, 0]], dtype=int)
    try:
        student = student_sol(A2, B2, C2, D2)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A2, B2, C2, D2)
    if (student != master):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    A3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    B3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    C3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    D3 = np.array([[0, 0], [0, 0], [0, 0]], dtype=int)
    try:
        student = student_sol(A3, B3, C3, D3)
    except Exception as e:
        print(f"Error: {e}")
        return False
    master = master_sol(A3, B3, C3, D3)
    if (student != master):
        print("Test failed: The transfer function is incorrect.")
        print("Student's result: ", student)
        print("Expected result: ", master)
    else:
        passed_tests += 1

    print("Passed tests:", passed_tests, " out of 3")
    return passed_tests == 3


def sol_calc_tf_ctrl(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, outputnumber: int = 0) -> Tuple[
    np.ndarray, np.ndarray, Optional[float]]:
    """
    Calculate the transfer function of the system using signal library.

    Parameters:
    - ``A`` (np.ndarray): The A matrix of the system.
    - ``B`` (np.ndarray): The B matrix of the system.
    - ``C`` (np.ndarray): The C matrix of the system.
    - ``D`` (np.ndarray): The D matrix of the system.

    Returns:
    - Tuple[np.ndarray, np.ndarray, Optional[float]]: The transfer function of the system.
    """
    # Check dimensions of A, B, C, D
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "The number of rows in A and B must be the same"
    assert A.shape[1] == C.shape[1], "The number of columns in A and C must be the same"
    assert B.shape[1] == D.shape[1] and C.shape[0] == D.shape[0], "D must be of dimensions (C.shape[0], B.shape[1])"

    transferfunction = signal.ss2tf(A, B, C, D, outputnumber)

    return transferfunction


def test_calc_tf_ctrl(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit tests for checking the implementation of calc_tf_ctrl(). This function should calculate the MIMO transfer function given A, B, C, and D matrices.
    """

    passed_tests = 0
    A1 = np.array([[5, 6, 6], [0, 0, 1], [-5, -5, -6]])
    B1 = np.array([[1, 2], [5, 6], [3, 4]])
    C1 = np.array([[1, -2, 1]])
    D1 = np.array([[1, 3]])
    num_outputs = 1
    for i in range(num_outputs):
        try:
            student = student_sol(A1, B1, C1, D1, i)
        except Exception as e:
            print(f"Error: {e}")
            return False
        master = master_sol(A1, B1, C1, D1, i)
        if (compare_tuples(student, master) == False):
            print("Test failed: The transfer function for output {i} is incorrect.")
            print("Student's result: ", student)
            print("Expected result: ", master)
        else:
            passed_tests += 1
    print("First test series passed")

    A2 = np.array([[0, 3], [0, 0]], dtype=int)
    B2 = np.array([[0, 0], [0, 1]], dtype=int)
    C2 = np.array([[1, 0], [0, 1]], dtype=int)
    D2 = np.array([[0, 0], [0, 0]], dtype=int)
    num_outputs = 2
    for i in range(num_outputs):
        try:
            student = student_sol(A2, B2, C2, D2, i)
        except Exception as e:
            print(f"Error: {e}")
            return False
        master = master_sol(A2, B2, C2, D2, i)
        if (compare_tuples(student, master) == False):
            print("Test failed: The transfer function for output {i} is incorrect.")
            print("Student's result: ", student)
            print("Expected result: ", master)
        else:
            passed_tests += 1
    print("Second test series passed")

    A3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    B3 = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], dtype=int)
    C3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    D3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    num_outputs = 3
    for i in range(num_outputs):
        try:
            student = student_sol(A3, B3, C3, D3, i)
        except Exception as e:
            print(f"Error: {e}")
            return False
        master = master_sol(A3, B3, C3, D3, i)
        if (compare_tuples(student, master) == False):
            print("Test failed: The transfer function is incorrect.")
            print("Student's result: ", student)
            print("Expected result: ", master)
        else:
            passed_tests += 1
    print("Third test series passed")

    print("Passed tests:", passed_tests, " out of 6")
    return passed_tests == 6


def compare_tuples(t1: Tuple[np.ndarray, np.ndarray, Optional[float]],
                   t2: Tuple[np.ndarray, np.ndarray, Optional[float]]) -> bool:
    return np.array_equal(t1[0], t2[0]) and np.array_equal(t1[1], t2[1])


def sol_get_tf(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[
    np.ndarray, signal.TransferFunction, signal.TransferFunction, signal.TransferFunction, signal.TransferFunction]:
    """
    Find the individual transfer functions in the transfer function matrix.

    Parameters:
    - ``A`` (np.ndarray): The A matrix of the system.
    - ``B`` (np.ndarray): The B matrix of the system.
    - ``C`` (np.ndarray): The C matrix of the system.
    - ``D`` (np.ndarray): The D matrix of the system.

    Returns:
    - Tuple[np.ndarray, TransferFunction, TransferFunction, TransferFunction, TransferFunction]: The individual transfer functions
    """
    # Check dimensions of A, B, C, D
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert A.shape[0] == B.shape[0], "The number of rows in A and B must be the same"
    assert A.shape[1] == C.shape[1], "The number of columns in A and C must be the same"
    assert B.shape[1] == D.shape[1] and C.shape[0] == D.shape[0], "D must be of dimensions (C.shape[0], B.shape[1])"

    tf_output1 = sol_calc_tf_ctrl(A, B, C, D, 0)
    tf_output2 = sol_calc_tf_ctrl(A, B, C, D, 1)

    den = tf_output1[1]

    tf_11 = signal.TransferFunction(tf_output1[0][0], den)
    tf_12 = signal.TransferFunction(tf_output2[0][0], den)
    tf_21 = signal.TransferFunction(tf_output1[0][1], den)
    tf_22 = signal.TransferFunction(tf_output2[0][1], den)

    return den, tf_11, tf_12, tf_21, tf_22
