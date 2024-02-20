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

def sol_aircraft_state_space() -> ct.StateSpace:
    """
    Produce the linear state space model of the aircraft.
  
    Parameters:
    - None

    Returns:
    - () -> StateSpace: The state space model of the aircraft.
    """
    A_sol = np.array([
    [-15.5801, 4.7122, -38.7221, 0],
    [-0.5257, -0.0166, 2.3501, -9.7847],
    [4.4044, -1.5325, -18.1615, -0.7044],
    [0.9974, 0, 0, 0]
    ])

    B_sol = np.array([
        [-421.2001],
        [1.3231],
        [-17.3812],
        [0]
    ])

    C_sol = np.array([[0, 0, 0, 1]])

    D_sol = np.array([[0]])

    return ct.StateSpace(A_sol, B_sol, C_sol, D_sol)


def test_aircraft_state_space(student_sol, actual_sol, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's state space model with the solution state space model.

    Parameters:
    - ``student_sol`` (function): A function that returns the student's state space model.
    - ``actual_sol`` (function): A function that returns the solution state space model.
    - ``shouldprint`` (bool): A boolean indicating whether to print the state space models. Default is True.

    Raises:
    - AssertionError: If the input is not an instance of the StateSpace class.

    Returns:
    - () -> bool: A boolean indicating whether the student's state space model is equal to the solution state space model.
    """

    # Create a state space model using the student's code
    student_model = student_sol()
    # Create a state space model using the solution code
    sol_model = actual_sol()

    # Check if the student's state space model is an instance of the StateSpace class
    assert isinstance(student_model, None), f"Please make sure to return a StateSpace object. Got {type(student_model)} instead."
    assert isinstance(student_model, ct.StateSpace), f"Expected a StateSpace object, but got {type(student_model)}"

    # Check if the student's state space model is equal to the solution state space model
    if shouldprint:
        print("Student's state space model:")
        print(student_model)
        print("\nSolution state space model:")
        print(sol_model)

    assert np.array_equal(student_model.A, sol_model.A), "The state space model A matrix is incorrect."
    assert np.array_equal(student_model.B, sol_model.B), "The state space model B matrix is incorrect."
    assert np.array_equal(student_model.C, sol_model.C), "The state space model C matrix is incorrect."
    assert np.array_equal(student_model.D, sol_model.D), "The state space model D matrix is incorrect."

    return True


def sol_is_system_stable(sys: ct.StateSpace) -> bool:
    """
    Determine if the given state space system is stable.

    Parameters:
    - ``sys`` (StateSpace): The state space model of the system.

    Returns:
    - () -> bool: A boolean indicating whether the system is stable.
    """
    return np.all(np.linalg.eigvals(sys.A) < 0)


def test_is_system_stable(student_sol, actual_sol, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's stability check with the solution stability check.

    Parameters:
    - ``student_sol`` (function): A function that returns the student's stability check.
    - ``actual_sol`` (function): A function that returns the solution stability check.
    - ``shouldprint`` (bool): A boolean indicating whether to print the stability checks. Default is True.

    Returns:
    - () -> bool: A boolean indicating whether the student's stability check is equal to the solution stability check.
    """

    passed_tests = 0

    # Conduct four unit tests
    # Test 0: Check if is_system_stable doesn't return None
    sys_dummy = ct.StateSpace(np.array([[1, 0], [0, 1]]), np.array([[1], [0]]), np.array([[0, 1]]), np.array([[0]]))
    result = student_sol(sys_dummy)
    print("Test failed: is_system_stable should return a boolean value." if result is None else "")
    print("Student's result: ", result)
    print("Expected result: ", actual_sol(sys_dummy))
    passed_tests += 1 if result == actual_sol(sys_dummy) else 0

    # Test 1: Check if is_system_stable returns False for an unstable system
    sys_unstable = ct.StateSpace(np.array([[1, 1], [0, 1]]), np.array([[1], [0]]), np.array([[1, 0]]), np.array([[0]]))
    result = student_sol(sys_unstable)
    print("Student's result: ", result)
    print("Expected result: ", actual_sol(sys_unstable))
    passed_tests += 1 if result == actual_sol(sys_unstable) else 0

    # Test 2: Marginally stable system
    sys_marginally_stable = ct.StateSpace(np.array([[0, 1], [-1, 0]]), np.array([[1], [0]]), np.array([[0, 1]]), np.array([[0]]))
    result = student_sol(sys_unstable)
    print("Student's result: ", result)
    print("Expected result: ", actual_sol(sys_marginally_stable))
    passed_tests += 1 if result == actual_sol(sys_marginally_stable) else 0

    # Test 3: Stable system
    sys_stable = ct.StateSpace(np.array([[-0.5, 0], [0, -1]]), np.array([[1], [0]]), np.array([[0, 1]]), np.array([[0]]))
    result = student_sol(sys_unstable)
    print("Student's result: ", result)
    print("Expected result: ", actual_sol(sys_stable))
    passed_tests += 1 if result == actual_sol(sys_stable) else 0

    # Test 4: Check aircraft stability
    sys_aircraft = sol_aircraft_state_space()
    result = student_sol(sys_aircraft)
    print("Expected result: ", actual_sol(sys_aircraft))
    passed_tests += 1 if result == actual_sol(sys_aircraft) else 0

    print("Passed tests: ", passed_tests, "/5")
    return passed_tests == 5


def sol_plot_initial_conditions_response(sys: ct.StateSpace, initial_conditions: np.array) -> None:
    """
    Plot the system response to the given initial conditions.
    
    Parameters:
    - ``sys`` (ct.StateSpace): The state space representation of the system.
    - ``initial_conditions`` (np.array): The initial conditions of the system.

    Returns:
        None
    """
    
    # Simulate the system response to the given initial conditions
    print(initial_conditions)
    time, response = ct.initial_response(sys, X0 = initial_conditions, T=250)

    # Plot the response
    plt.figure(figsize=(10, 6))

    # Plot pitch rate (q)
    plt.plot(time, response)
    plt.xlabel('Time (s)')
    plt.ylabel('System output')
    plt.title('Initial condition response')
    plt.show()

    return None