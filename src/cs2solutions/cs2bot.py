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

def sol_get_xdot(t: float, x: np.array, u: np.array, params: dict) -> np.array:
    """
    Solution: Return the derivative of the state vector, aka x\dot.

    Parameters:
    - ``t`` (float): The current time.	
    - ``x`` (np.array): The current state.
    - ``u`` (np.array): The current input.
    - ``params`` (dict): A dictionary of parameters.

    Returns:
    - np.array: The derivative of the state.
    """

     # Return the derivative of the state
    max_omega = params.get('max_omega', 0.05)
    omega = np.clip(u[1], -max_omega, max_omega)
    return np.array([
        u[0] * np.cos(x[2]),    # xdot = v cos(theta)
        u[0] * np.sin(x[2]),    # ydot = v sin(theta)
        omega     # thdot = w
    ])

def assert_almost_equal(actual, expected, tolerance=1e-5) -> bool:
    try:
        almost_equal = np.all(np.abs(actual - expected) < tolerance)
        return almost_equal
    except Exception:
        return False

def test_xdot(student_sol, actual_sol, tolerance: float = 1e-5) -> bool:
    """
    Tests the student solution for get_xdot.

    Parameters:
    - ``student_sol`` (function): Student solution.	
    - ``actual_sol`` (function): Expected solution function.
    - ``tolerance`` (float): Tolerance for float equality.

    Returns:
    - bool: Whether all unit tests passed.
    """
    passed_tests = 0

    # Test 0: Check correct return value
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])   # initial state [x, y, theta]
    u = np.array([1.0, 0.1])        # input [v, omega]
    result = student_sol(0.0, x, u, params)
    solution = actual_sol(0.0, x, u, params)
    print("Test failed: get_xdot should a np.array" if result is None else "")
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    # Test 1: Check forward motion
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])  # initial state [x, y, theta]
    u = np.array([2.0, 0.0])       # input [v, omega]
    result = student_sol(0.0, x, u, params)
    solution = actual_sol(0.0, x, u, params)
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    # Test 2: Check rotation
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])
    u = np.array([0.0, 0.01])
    result = student_sol(0.0, x, u, params)
    solution = np.array([0.0, 0.0, 0.01])
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    # Test 3: Check clipping omega
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])
    u = np.array([0.0, 0.1])  # omega exceeds max_omega
    result = student_sol(0.0, x, u, params)
    solution = np.array([0.0, 0.0, params['max_omega']])
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    # Test 4: Check combined motion
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, np.pi/4])
    u = np.array([1.0, 0.1])
    result = student_sol(0.0, x, u, params)
    solution = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.05])
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    print("Passed tests: ", passed_tests, "/5")
    return passed_tests == 5


def sol_get_y(t: float, x: np.array, u: np.array, params: dict) -> np.array:
    """
    Solution: Returns the state vector, aka [x, y, \omega].

    Parameters:
    - ``t`` (float): The current time.	
    - ``x`` (np.array): The current state.
    - ``u`` (np.array): The current input.
    - ``params`` (dict): A dictionary of parameters.

    Returns:
    - np.array: The the state vector
    """
    # Returns the state vector
    return x[0:2]


def test_y(student_sol, actual_sol, tolerance: float = 1e-5) -> bool:
    """
    Tests the student solution for get_y.

    Parameters:
    - ``student_sol`` (function): Student solution.	
    - ``actual_sol`` (function): Expected solution function.
    - ``tolerance`` (float): Tolerance for float equality.

    Returns:
    - bool: Whether all unit tests passed.
    """
    passed_tests = 0

    # Test 0: Check correct return value
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])   # initial state [x, y, theta]
    u = np.array([1.0, 0.1])        # input [v, omega]
    result = student_sol(0.0, x, u, params)
    solution = actual_sol(0.0, x, u, params)
    print("Test failed: get_y should return a np.array" if result is None else "")
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    # Test 1: Check forward motion
    params = {'max_omega': 0.05}
    x = np.array([1.0, 2.0, 0.0])  # initial state [x, y, theta]
    u = np.array([2.0, 0.0])       # input [v, omega]
    result = student_sol(0.0, x, u, params)
    solution = np.array([1.0, 2.0])
    print("Student's result: ", result)
    print("Expected result: ", solution)
    passed_tests += 1 if assert_almost_equal(result, solution, tolerance) else 0

    print("Passed tests: ", passed_tests, "/2")
    return passed_tests == 2
