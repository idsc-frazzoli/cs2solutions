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


def print_nice(state: List[np.ndarray]) -> None:
    """
    Prints the state-space matrices in a nice format.

    Parameters:
    - ``state`` (List[np.ndarray]): A list of state-space matrices.
    """
    matrix_names = ["A", "B", "C", "D"]
    for mat, name in zip(state, matrix_names):
        print(f"Matrix {name}:")
        print(np.array2string(mat, precision=3, suppress_small=True))
        print()

def sol_diagonalize_system(A: np.array, B: np.array, C: np.array, D: np.array) -> tuple[np.array]:
    """
    Diagonalizes a linear time-invariant system represented by its state-space matrices.

    Parameters:
    - ``A`` (np.array): State matrix of the system.
    - ``B`` (np.array): Input matrix of the system.
    - ``C`` (np.array): Output matrix of the system.
    - ``D`` (np.array): Feedthrough matrix of the system.

    Returns:
    - ``A_tilde`` (np.array): Diagonalized state matrix.
    - ``B_tilde`` (np.array): Diagonalized input matrix.
    - ``C_tilde`` (np.array): Diagonalized output matrix.
    - ``D_tilde`` (np.array): Diagonalized feedthrough matrix.
    """
    # Preliminary safety checks
    D = np.array(D)
    if not all(isinstance(mat, np.ndarray) for mat in [A, B, C, D]):
        raise TypeError("All input matrices must be numpy arrays.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    # Compute the eigenvalues and eigenvectors of matrix A
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Compute the transformed state-space representation
    # The diagonal matrix A_tilde contains the eigenvalues of A
    A_tilde = np.diag(eigenvalues)
    
    # The input matrix B_tilde is obtained by transforming the input matrix B
    # using the inverse of the eigenvector matrix
    B_tilde = np.linalg.inv(eigenvectors) @ B
    
    # The output matrix C_tilde is obtained by transforming the output matrix C
    # using the eigenvector matrix
    C_tilde = C @ eigenvectors
    
    # The feedthrough matrix D_tilde remains the same
    D_tilde = D
    
    return A_tilde, B_tilde, C_tilde, D_tilde


def test_diagonalize_system(student_sol: callable, actual_sol: callable, A3: np.array, B3: np.array, C3: np.array, D3: np.array, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's diagonalized system with the solution diagonalized system.

    Parameters:
    - ``student_sol`` (function): A function that returns the student's diagonalized system.
    - ``actual_sol`` (function): A function that returns the solution diagonalized system.
    - ``A3`` (np.array): State matrix of the system.
    - ``B3`` (np.array): Input matrix of the system.
    - ``C3`` (np.array): Output matrix of the system.
    - ``D3`` (np.array): Feedthrough matrix of the system.
    - ``shouldprint`` (bool): A boolean indicating whether to print the diagonalized systems. Default is True.

    Raises:
    - AssertionError: If the input matrices are not numpy arrays.

    Returns:
    - () -> bool: A boolean indicating whether the student's diagonalized system is equal to the solution diagonalized system.
    """
    passed_tests = 0

    # Already diagonalized system
    A1 = np.array([[1, 0], [0, 2]])
    B1 = np.array([[1], [1]])
    C1 = np.array([[1, 0]])
    D1 = np.array([[0]])
    try:
        student1 = student_sol(A1, B1, C1, D1)
    except Exception as e:
        print("Error in diagonalize_system:", e)
        student1 = None
    solution1 = actual_sol(A1, B1, C1, D1)
    if shouldprint: print("Student 1:")
    if shouldprint: print_nice(student1)
    if shouldprint: print("Solution 1:")
    if shouldprint: print_nice(solution1)
    passed_tests += 1 if all(np.allclose(a, b) for a, b in zip(student1, solution1)) else 0

    # Random system
    A2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-2, 1, 0, 0], [1, -1, 0, 0]]);
    B2 = np.array([[0], [0], [1], [0]])
    C2 = np.array([1, 0, 0, 0])
    D2 = np.array([0])
    try:
        student2 = student_sol(A2, B2, C2, D2)
    except Exception as e:
        print("Error in diagonalize_system:", e)
        student2 = None
    solution2 = actual_sol(A2, B2, C2, D2)
    if shouldprint: print("Student 2:")
    if shouldprint: print_nice(student2)
    if shouldprint: print("Solution 2:")
    if shouldprint: print_nice(solution2)
    passed_tests += 1 if all(np.allclose(a, b) for a, b in zip(student2, solution2)) else 0

    # Aircraft System
    D3 = np.array(D3)
    try:
        student3 = student_sol(A3, B3, C3, D3)
    except Exception as e:
        print("Error in diagonalize_system:", e)
        student3 = None
    solution3 = actual_sol(A3, B3, C3, D3)
    if shouldprint: print("Student 3:")
    if shouldprint: print_nice(student3)
    if shouldprint: print("Solution 3:")
    if shouldprint: print_nice(solution3)
    passed_tests += 1 if all(np.allclose(a, b) for a, b in zip(student3, solution3)) else 0

    print("Passed tests:", passed_tests, " out of 3")
    return passed_tests == 3


def sol_controllable(A: np.array, B: np.array) -> bool:
    """
    Check the controllability of a system given the state-space representation.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``B`` (np.array): The input matrix.

    Returns:
    - () -> bool: A boolean indicating whether the system is controllable.
    """
    
    n = A.shape[0]  # Get the number of rows in matrix A.
    R = B  # Initialize the controllability matrix R with matrix B, since the first column of R is always B.
    
    for i in range(1, n):  # Iterate from 1 to n-1.
        # Calculate the reachability matrix for each power of A and concatenate it horizontally to R.
        R = np.hstack((R, np.linalg.matrix_power(A, i) @ B))
    
    rank = np.linalg.matrix_rank(R)
    
    if rank == A.shape[0]:
        print(f"The system is controllable with rank {rank}.")
    else:
        print(f"The system is not controllable with rank {rank}.")
    return rank == n


def sol_observable(A: np.array, C: np.array) -> None:
    """
    Check the observability of a system given the state-space representation.

    Parameters:
    - ``A`` (np.array): The state matrix.
    - ``C`` (np.array): The output matrix.

    Returns:
    - () -> bool: A boolean indicating whether the system is observable.
    """

    n = A.shape[0]  # Get the number of rows in matrix A.
    O = C  # Initialize the observability matrix O with matrix C, since the first row of O is always C.
    
    """
    Same as above, maybe show how to extract the first row programmatically (e.g. O = C[0,:])
    """
    
    for i in range(1, n):  # Iterate from 1 to n-1.
        # Calculate the observability matrix for each power of A and concatenate it vertically to O.
        O = np.vstack((O, C @ np.linalg.matrix_power(A, i)))
    
    rank = np.linalg.matrix_rank(O)
    
    if rank == A.shape[0]:
        print(f"The system is observable with rank {rank}.")
    else:
        print(f"The system is not observable with rank {rank}.")
    return rank == n


def test_controllable(student_sol: callable, actual_sol: callable, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's controllability check with the solution controllability check.

    Parameters:
    - ``student_sol`` (function): A function that returns the student's controllability check.
    - ``actual_sol`` (function): A function that returns the solution controllability check.
    - ``shouldprint`` (bool): A boolean indicating whether to print the controllability checks. Default is True.

    Returns:
    - () -> bool: A boolean indicating whether the student's controllability check is equal to the solution controllability check.
    """
    passed_tests = 0

    # Controllable system
    A1 = np.array([[1, 0], [0, 2]])
    B1 = np.array([[1], [1]])
    print("Student solution:")
    try:
        student1 = student_sol(A1, B1)
        print(student1)
    except Exception as e:
        print("Error in controllable:", e)
        student1 = None
    print("Master solution:")
    solution1 = actual_sol(A1, B1)
    print(solution1)
    passed_tests += 1 if student1 == solution1 else 0

    # Controllable system
    A2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-2, 1, 0, 0], [1, -1, 0, 0]])
    B2 = np.array([[0], [0], [1], [0]])
    print("Student solution:")
    try:
        student2 = student_sol(A2, B2)
        print(student2)
    except Exception as e:
        print("Error in controllable:", e)
        student2 = None
    print("Master solution:")
    solution2 = actual_sol(A2, B2)
    print(solution2)
    passed_tests += 1 if student2 == solution2 else 0

    print("Passed tests:", passed_tests, " out of 2")
    return passed_tests == 2


def test_observable(student_sol: callable, actual_sol: callable, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's observability check with the solution observability check.

    Parameters:
    - ``student_sol`` (function): A function that returns the student's observability check.
    - ``actual_sol`` (function): A function that returns the solution observability check.
    - ``shouldprint`` (bool): A boolean indicating whether to print the observability checks. Default is True.

    Returns:
    - () -> bool: A boolean indicating whether the student's observability check is equal to the solution observability check.
    """
    passed_tests = 0

    # Observable system
    A1 = np.array([[1, 0], [0, 2]])
    C1 = np.array([[1, 1]])
    print("Student solution:")
    try:
        student1 = student_sol(A1, C1)
        print(student1)
    except Exception as e:
        print("Error in observable:", e)
        student1 = None
    print("Master solution:")
    solution1 = actual_sol(A1, C1)
    print(solution1)
    passed_tests += 1 if student1 == solution1 else 0

    # Unobservable system
    A2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-2, 1, 0, 0], [1, -1, 0, 0]])
    C2 = np.array([[0, 0, 0, 0]])
    print("Student solution:")
    try:
        student2 = student_sol(A2, C2)
        print(student2)
    except Exception as e:
        print("Error in observable:", e)
        student2 = None
    print("Master solution:")
    solution2 = actual_sol(A2, C2)
    print(solution2)
    passed_tests += 1 if student2 == solution2 else 0

    print("Passed tests:", passed_tests, " out of 2")
    return passed_tests == 2