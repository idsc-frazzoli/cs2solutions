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
    if shouldprint: print("Student 1:", student1)
    if shouldprint: print("Solution 1:", solution1)
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
    if shouldprint: print("Student 2:", student2)
    if shouldprint: print("Solution 2:", solution2)
    passed_tests += 1 if all(np.allclose(a, b) for a, b in zip(student2, solution2)) else 0

    # Aircraft System
    try:
        student3 = student_sol(A3, B3, C3, D3)
    except Exception as e:
        print("Error in diagonalize_system:", e)
        student3 = None
    solution3 = actual_sol(A3, B3, C3, D3)
    if shouldprint: print("Student 3:", student3)
    if shouldprint: print("Solution 3:", solution3)
    passed_tests += 1 if all(np.allclose(a, b) for a, b in zip(student3, solution3)) else 0

    return passed_tests == 3







