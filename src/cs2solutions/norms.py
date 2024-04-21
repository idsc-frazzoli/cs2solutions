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

def sol_vector2norm(vector: np.ndarray) -> float:
    """
    Calculate the 2-norm (Euclidean norm) of a NumPy array.

    Parameters:
    vector (np.ndarray): Input array for which 2-norm is to be calculated.

    Returns:
    float: The 2-norm of the input array.
    """
    if len(vector) == 0 or vector is None or not isinstance(vector, np.ndarray):
        raise ValueError("Input vector is invalid")
    
    squared_sum = 0.0
    for elem in vector:
        squared_sum += elem**2
    norm = np.sqrt(squared_sum)

    return norm

def sol_vectorInfnorm(vector: np.ndarray) -> float:
    """
    Calculate the infinity-norm (maximum absolute value) of a NumPy array.

    Parameters:
    vector (np.ndarray): Input array for which infinity-norm is to be calculated.

    Returns:
    float: The infinity-norm of the input array.
    """
    if len(vector) == 0 or vector is None or not isinstance(vector, np.ndarray):
        raise ValueError("Input vector is invalid")
    
    max_elem = 0.0
    for elem in vector:
        if abs(elem) > max_elem:
            max_elem = abs(elem)

    return max_elem

def sol_vectorPnorm(vector: np.ndarray, p: float) -> float:
    """
    Calculate the p-norm (maximum absolute value) of a NumPy array.

    Parameters:
    vector (np.ndarray): Input array for which p-norm is to be calculated.

    Returns:
    float: The p-norm of the input array.
    """
    if len(vector) == 0 or vector is None or not isinstance(vector, np.ndarray):
        raise ValueError("Input vector is invalid")

    if p is None:
        return sol_vector2norm(vector)

    p = float(p)
    if not isinstance(p, float) or p <= 0.0:
        raise ValueError("p must be a positive integer")

    norm_sum = 0.0
    for elem in vector:
        norm_sum += abs(elem)**p
    
    return norm_sum**(1/p)

def test_vector_norm(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit test function to see if the student solution for 'vector2norm' or 'vectorInfnorm' is correct.

    Parameters:
    - ``student_sol`` (callable): The student's solution function.
    - ``master_sol`` (callable): The master solution function.
    - ``shouldprint`` (bool): Flag to print the test results.

    Returns:
    bool: The test result.
    """

    vectors = [
        np.array([1, 2, 3, 4, 5]),
        np.array([0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1]),
        np.array([1, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0]),
        np.array([2, 4, 3.5, 2, 1.5]),
        np.array([-1, -2, -3, -4, -5]),
        np.array([1, -2, 3, -4, 5]),
        np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        np.array([1, 1, 1, 1, 100])
    ]

    passed_tests = 0

    for i, v in enumerate(vectors):
        try: 
            student_result = student_sol(v)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        master_result = master_sol(v)
        correct_answer = master_result == student_result

        if shouldprint and not correct_answer: 
            print("Error in vector ", i, ": ", v)
            print("Student's result: ", student_result)
            print("Expected result: ", master_result)

        passed_tests += 1 if correct_answer else 0

    print("Passed tests: ", passed_tests, " out of 10")
    return passed_tests == 10


def test_vectorPnorm(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit test function to see if the student solution for 'vectorPnorm' is correct.

    Parameters:
    - ``student_sol`` (callable): The student's solution function.
    - ``master_sol`` (callable): The master solution function.
    - ``shouldprint`` (bool): Flag to print the test results.

    Returns:
    bool: The test result.
    """

    vectors = [
        np.array([1, 2, 3, 4, 5]),
        np.array([0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1]),
        np.array([1, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0]),
        np.array([2, 4, 3.5, 2, 1.5]),
        np.array([-1, -2, -3, -4, -5]),
        np.array([1, -2, 3, -4, 5]),
        np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        np.array([1, 1, 1, 1, 100])
    ]
    p_values = [1, 2, 3, 4, 5]
    passed_tests = 0
    one_incorrect = False
    for i, v in enumerate(vectors):
        for p in p_values:
            try: 
                student_result = student_sol(v, p)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            master_result = master_sol(v, p)
            correct_answer = master_result == student_result

            if shouldprint and not correct_answer and not one_incorrect: 
                print("Error in vector ", i, ": ", v, " with p = ", p)
                print("Student's result: ", student_result)
                print("Expected result: ", master_result)
                one_incorrect = True

            passed_tests += 1 if correct_answer else 0

    print("Passed tests: ", passed_tests, " out of 50")
    return passed_tests == 50

def sol_matrix2norm(matrix: np.ndarray) -> float:
    """
    Calculate the 2-norm of a matrix manually.

    Parameters:
    matrix (numpy.ndarray): Input matrix for which the 2-norm is to be calculated.

    Returns:
    float: The 2-norm of the input matrix.
    """
    if matrix is None or not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix is invalid")

    product_matrix = np.dot(matrix.T, matrix)
    eigenvalues, _ = np.linalg.eig(product_matrix)
    max_eigenvalue = np.max(eigenvalues)
    norm = np.sqrt(max_eigenvalue)
    return norm

def sol_calculate_matrix_singular_values(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the singular values of a matrix manually.

    Parameters:
    matrix (numpy.ndarray): Input matrix for which the singular values are to be calculated.

    Returns:
    numpy.ndarray: The singular values of the input matrix.
    """
    if matrix is None or not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix is invalid")

    product_matrix = np.dot(matrix.T, matrix) if matrix.shape[0] > matrix.shape[1] else np.dot(matrix, matrix.T)
    eigenvalues, _ = np.linalg.eig(product_matrix)
    singular_values = np.sqrt(np.abs(eigenvalues))
    return singular_values

def sol_matrixPnorm(matrix: np.ndarray, p: float) -> float:
    """
    Calculate the p-norm of a matrix manually.

    Parameters:
    matrix (numpy.ndarray): Input matrix for which the p-norm is to be calculated.

    Returns:
    float: The p-norm of the input matrix.
    """
    if matrix is None or not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix is invalid")
    if p is None:
        return sol_matrix2norm(matrix)
    p = float(p)
    if not isinstance(p, float) or p <= 0.0:
        raise ValueError("p must be a positive integer")

    singular_values = sol_calculate_matrix_singular_values(matrix)
    norm = np.sum(np.abs(singular_values)**p)**(1.0/p)
    return norm

def sol_matrixInfnorm(matrix: np.ndarray) -> float:
    """
    Calculate the infinity-norm of a matrix manually.

    Parameters:
    matrix (numpy.ndarray): Input matrix for which the infinity-norm is to be calculated.

    Returns:
    float: The infinity-norm of the input matrix.
    """
    if matrix is None or not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix is invalid")

    row_sums = np.sum(np.abs(matrix), axis=1)  # Calculate absolute row sums
    norm = np.max(row_sums)  # Take the maximum of the absolute row sums
    return norm

def test_matrix_norm(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit test function to see if the student solution for 'matrix2norm' or 'matrixInfnorm' is correct.

    Parameters:
    - ``student_sol`` (callable): The student's solution function.
    - ``master_sol`` (callable): The master solution function.
    - ``shouldprint`` (bool): Flag to print the test results.

    Returns:
    bool: The test result.
    """

    matrices = [
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[2, 4], [3.5, 2]]),
        np.array([[-1, -2], [-3, -4]]),
        np.array([[1, -2], [3, -4]]),
        np.array([[1.5, 2.5], [3.5, 4.5]]),
        np.array([[1, 1], [1, 100]])
    ]

    passed_tests = 0

    for i, m in enumerate(matrices):
        try: 
            student_result = student_sol(m)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        master_result = master_sol(m)
        correct_answer = master_result == student_result

        if shouldprint and not correct_answer: 
            print("Error in matrix ", i, ": ", m)
            print("Student's result: ", student_result)
            print("Expected result: ", master_result)

        passed_tests += 1 if correct_answer else 0

    print("Passed tests: ", passed_tests, " out of 10")
    return passed_tests == 10

def test_matrixPnorm(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Unit test function to see if the student solution for 'matrixPnorm' is correct.

    Parameters:
    - ``student_sol`` (callable): The student's solution function.
    - ``master_sol`` (callable): The master solution function.
    - ``shouldprint`` (bool): Flag to print the test results.

    Returns:
    bool: The test result.
    """

    matrices = [
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[2, 4], [3.5, 2]]),
        np.array([[-1, -2], [-3, -4]]),
        np.array([[1, -2], [3, -4]]),
        np.array([[1.5, 2.5], [3.5, 4.5]]),
        np.array([[1, 1], [1, 100]])
    ]
    p_values = [1, 2, 3, 4, 5]
    passed_tests = 0
    one_incorrect = False
    for i, m in enumerate(matrices):
        for p in p_values:
            try: 
                student_result = student_sol(m, p)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            master_result = master_sol(m, p)
            correct_answer = master_result == student_result

            if shouldprint and not correct_answer and not one_incorrect: 
                print("Error in vector ", i, ": ", m, " with p = ", p)
                print("Student's result: ", student_result)
                print("Expected result: ", master_result)
                one_incorrect = True

            passed_tests += 1 if correct_answer else 0

    print("Passed tests: ", passed_tests, " out of 50")
    return passed_tests == 50