from typing import List, Tuple

try:
    from typing import Optional, List, Tuple
    import numpy as np
    import matplotlib.pyplot as plt
    import control as ct
    import scipy.linalg as la
    from scipy import signal as sig
    from scipy.signal import butter, lfilter
    import unittest as unit
except ImportError as e:
    print(f"Error: {e}")
    print(f"Please install the required packages using the command '!pip install control numpy matplotlib scipy'")

def sol_H_inf_norm_state_space(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> float:
    """
    Returns the infinity norm of the system defined by the state space matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``C`` (np.ndarray): The output matrix of the system
    - ``D`` (np.ndarray): The feedthrough matrix of the system

    Returns:
    - float: The infinity norm of the system
    """
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert B.shape[0] == A.shape[0], "Matrix B must have the same number of rows as A"
    assert C.shape[1] == A.shape[1], "Matrix C must have the same number of columns as A"
    assert D.shape[0] == C.shape[0] and D.shape[1] == B.shape[1], "Matrix D must have the same number of rows as C and the same number of columns as B"

    gamma, step, precision = 1, 1, 1
    
    while precision > 1e-7:
        # Calculate the Hamiltonian matrix
        H = np.block([[A, 1/gamma*B@B.T], [-1/gamma*C.T@C, -A.T]])

        # Check if the eigenvalues are on the imaginary axis, then update gamma
        if any(np.isclose(np.linalg.eigvals(H).real, 0)):
            prev_gamma = gamma
            gamma = gamma + step
        else:
            prev_gamma = gamma
            gamma = gamma - step
        
        precision = abs(gamma - prev_gamma)
        step /= 2
        if gamma == 0:
            gamma = 1e-7
    
    return gamma

def test_H_inf_norm_state_space(student_sol: callable, master_sol: callable, shouldprint: bool=True) -> bool:
    """
    Unit tests for the function ``H_inf_norm_state_space``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    test_cases = [
        {
            'A': np.array([[5, 6, 6], [0, 0, 1], [-5, -5, -6]]),
            'B': np.array([[1], [5], [3]]),
            'C': np.array([[1, -2, 1]]),
            'D': np.array([[1]])
        },
        {
            'A': np.array([[1, 2], [0, 1]]),
            'B': np.array([[1], [1]]),
            'C': np.array([[1, 0]]),
            'D': np.array([[0]])
        },
        {
            'A': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            'B': np.array([[1], [2], [1]]),
            'C': np.array([[1, 0, 0]]),
            'D': np.array([[0]])
        }
    ]

    passed_tests = 0
    total_tests = len(test_cases)
    for i, test_case in enumerate(test_cases):
        try:
            student = student_sol(test_case['A'], test_case['B'], test_case['C'], test_case['D'])
        except Exception as e:
            print(f"Could not run student solution for test case {i}: {e}")
            return False
        master = master_sol(test_case['A'], test_case['B'], test_case['C'], test_case['D'])
        if np.isclose(student, master):
            passed_tests += 1
        else: 
            if shouldprint:
                print(f"Test case {i} failed: student={student}, master={master}")
            

    print("Passed tests: ", passed_tests, "/", total_tests)
    return passed_tests == total_tests

def sol_solve_riccati_equation(A: np.ndarray, B_w: np.ndarray, C_y: np.ndarray, R_ww: np.ndarray, D_yw: np.ndarray) -> np.ndarray:
    """
    Returns the solution to the continuous algebraic Riccati equation

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B_w`` (np.ndarray): The disturbance input matrix of the system
    - ``C_y`` (np.ndarray): The output matrix of the system
    - ``R_ww`` (np.ndarray): The disturbance covariance matrix
    - ``D_yw`` (np.ndarray): The feedthrough matrix from the disturbance to the output
    - ``Q`` (np.ndarray): The state cost matrix

    Returns:
    - np.ndarray: The solution to the continuous algebraic Riccati equation
    """
    a = (A - B_w@D_yw.T@np.linalg.inv(R_ww)@C_y).T
    b = C_y
    q = B_w@(np.eye(A.shape[0])-D_yw.T@np.linalg.inv(R_ww)@D_yw)@B_w.T
    r = np.linalg.inv(R_ww)  
    
    # to combat floating point errors
    a = np.round(a, 10)
    b = np.round(b, 10)
    q = np.round(q, 10)
    r = np.round(r, 10)
    
    Y = la.solve_continuous_are(a, b, q, r)
    return Y

def sol_optimal_LQE(A: np.ndarray, B_w: np.ndarray, C_y: np.ndarray, D_yw: np.ndarray) -> np.ndarray:
    """
    Returns the optimal state feedback gain for the given system and cost matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``Q`` (np.ndarray): The state cost matrix
    - ``R`` (np.ndarray): The input cost matrix

    Returns:
    - np.ndarray: The optimal state feedback gain
    """
    R_ww = D_yw @ D_yw.T
    Y = sol_solve_riccati_equation(A, B_w, C_y, R_ww, D_yw)
    L = -(Y@C_y+B_w@D_yw.T)@np.linalg.inv(R_ww)
    
    return L

def test_solve_riccati_equation(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``solve_riccati_equation``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T

    try:
        student = student_sol(A, B_w, C_y, R_ww, D_yw)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, C_y, R_ww, D_yw)
    if np.allclose(student, master):
        print("Student implementation of solve_riccati_equation passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of solve_riccati_equation failed the test. Inspect ``sol_solve_riccati_equation`` to see the solution.")
            print(f"Student solution Y = : {student}")
            print(f"Master solution Y =: {master}")
        return False
    
def test_optimal_LQE(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``optimal_LQE``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T

    try:
        student = student_sol(A, B_w, C_y, D_yw)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, C_y, D_yw)
    if np.allclose(student, master):
        print("Student implementation of optimal_LQE passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of optimal_LQE failed the test. Inspect ``sol_optimal_LQE`` to see the solution.")
            print(f"Student solution L = : {student}")
            print(f"Master solution L =: {master}")
        return False
    
def sol_optimal_LQR(A: np.ndarray, B_u: np.ndarray, C_z: np.ndarray, D_zu: np.ndarray) -> np.ndarray:
    """
    Returns the optimal state feedback gain for the given system and cost matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``Q`` (np.ndarray): The state cost matrix
    - ``R`` (np.ndarray): The input cost matrix

    Returns:
    - np.ndarray: The optimal state feedback gain
    """
    a = A
    b = B_u
    q = C_z.T@C_z
    r = D_zu.T@D_zu
    
    # to combat floating point errors
    a = np.round(a, 10)
    b = np.round(b, 10)
    q = np.round(q, 10)
    r = np.round(r, 10)
    
    X_F = la.solve_continuous_are(a, b, q, r)
    F = -np.linalg.inv(r)@B_u.T@X_F
    
    return F

def test_optimal_LQR(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``optimal_LQR``.
    
    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T

    try: 
        student = student_sol(A, B_u, C_z, D_zu)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_u, C_z, D_zu)
    if np.allclose(student, master):
        print("Student implementation of optimal_LQR passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of optimal_LQR failed the test. Inspect ``sol_optimal_LQR`` to see the solution.")
            print(f"Student solution F = : {student}")
            print(f"Master solution F =: {master}")
        return False
    
def optimal_controller(A: np.ndarray, B_w: np.ndarray, B_u: np.ndarray, C_z: np.ndarray, C_y: np.ndarray, D_zw: np.ndarray, D_zu: np.ndarray, D_yw: np.ndarray, D_yu: np.ndarray) -> np.ndarray:
    """
    Returns the optimal state feedback gain for the given system and cost matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``Q`` (np.ndarray): The state cost matrix
    - ``R`` (np.ndarray): The input cost matrix

    Returns:
    - np.ndarray: The optimal state feedback gain
    """
    # Find the optimal observer gain
    L = sol_optimal_LQE(A, B_w, C_y, D_yw)
    
    # Find the optimal controller gain
    F = sol_optimal_LQR(A, B_u, C_z, D_zu)
    
    A_K = A+B_u@F+L@C_y
    B_K = np.block([[-L, B_u]])
    C_K = np.block([[F], [-C_y]])
    D_K = np.block([
        [np.zeros((F.shape[1], L.shape[0])), np.ones((F.shape[1], B_u.shape[1]))], 
        [np.ones((C_y.shape[0], L.shape[0])), np.zeros((C_y.shape[0], B_u.shape[1]))]])
    
    K = np.block([[A_K, B_K], [C_K, D_K]])
    
    return K, A_K, B_K, C_K, D_K

def test_optimal_controller(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``optimal_controller``.
    
    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T

    try:
        student = student_sol(A, B_w, B_u, C_z, C_y, D_zw, D_zu, D_yw, D_yu)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, B_u, C_z, C_y, D_zw, D_zu, D_yw, D_yu)
    if np.allclose(student[0], master[0]):
        print("Student implementation of optimal_controller passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of optimal_controller failed the test. Inspect ``optimal_controller`` to see the solution.")
            print(f"Student solution K = : {student[0]}")
            print(f"Master solution K =: {master[0]}")
        return False
    
def sol_find_X_inf(A: np.ndarray, B_w: np.ndarray, B_u: np.ndarray, C_z: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns the infinity norm of the system defined by the state space matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``C`` (np.ndarray): The output matrix of the system
    - ``D`` (np.ndarray): The feedthrough matrix of the system

    Returns:
    - float: The infinity norm of the system
    """
    a = A
    b = np.eye(A.shape[1])
    q = C_z.T@C_z
    r = np.linalg.inv(1/gamma**2*B_w@B_w.T-B_u@B_u.T)
    
    # to combat floating point errors
    a = np.round(a, 10)
    b = np.round(b, 10)
    q = np.round(q, 10)
    r = np.round(r, 10)
    
    X = la.solve_continuous_are(a, b, q, r)
    return X

def test_find_X_inf(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``find_X_inf``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T
    gamma = 0.5

    try:
        student = student_sol(A, B_w, B_u, C_z, gamma)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, B_u, C_z, gamma)
    if np.allclose(student, master):
        print("Student implementation of find_X_inf passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of find_X_inf failed the test. Inspect ``sol_find_X_inf`` to see the solution.")
            print(f"Student solution X = : {student}")
            print(f"Master solution X =: {master}")
        return False

def sol_find_Y_inf(A: np.ndarray, B_w: np.ndarray, C_z: np.ndarray, C_y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns the infinity norm of the system defined by the state space matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``C`` (np.ndarray): The output matrix of the system
    - ``D`` (np.ndarray): The feedthrough matrix of the system

    Returns:
    - float: The infinity norm of the system
    """
    a = A
    b = np.eye(A.shape[1])
    q = B_w.T@B_w
    r = np.linalg.inv(1/gamma**2*C_z@C_z.T-C_y@C_y.T)
    
    # to combat floating point errors
    a = np.round(a, 10)
    b = np.round(b, 10)
    q = np.round(q, 10)
    r = np.round(r, 10)
    
    Y = la.solve_continuous_are(a, b, q, r)
    return Y

def test_find_Y_inf(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``find_Y_inf``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T
    gamma = 0.5

    try:
        student = student_sol(A, B_w, C_z, C_y, gamma)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, C_z, C_y, gamma)
    if np.allclose(student, master):
        print("Student implementation of find_Y_inf passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of find_Y_inf failed the test. Inspect ``sol_find_Y_inf`` to see the solution.")
            print(f"Student solution Y = : {student}")
            print(f"Master solution Y =: {master}")
        return False

def assemble_suboptimal_K(A: np.ndarray, B_w: np.ndarray, B_u: np.ndarray, C_z: np.ndarray, C_y: np.ndarray, D_zw: np.ndarray, D_zu: np.ndarray, D_yw: np.ndarray, D_yu: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns the optimal state feedback gain for the given system and cost matrices

    Parameters:
    - ``A`` (np.ndarray): The state matrix of the system
    - ``B`` (np.ndarray): The input matrix of the system
    - ``Q`` (np.ndarray): The state cost matrix
    - ``R`` (np.ndarray): The input cost matrix

    Returns:
    - np.ndarray: The optimal state feedback gain
    """    
    # Find the suboptimal X_inf and Y_inf
    X_inf = sol_find_X_inf(A, B_w, B_u, C_z, gamma)
    Y_inf = sol_find_Y_inf(A, B_w, C_z, C_y, gamma)
    
    # Find the suboptimal gains
    F_inf = -B_u.T@X_inf
    L_inf = -Y_inf@C_y.T
    
    # Define helper matrices
    Z_inf = np.linalg.inv(np.eye(A.shape[0])-1/gamma**2*X_inf@Y_inf)
    
    # Assemble the suboptimal controller
    top_left = A + 1/gamma**2 *B_w@B_w.T@X_inf + B_u@F_inf + Z_inf@C_y
    top_right = -Z_inf@L_inf
    bottom_left = F_inf
    bottom_right = np.zeros((F_inf.shape[0], L_inf.shape[1]))
    
    K = np.block([[top_left, top_right], [bottom_left, bottom_right]])
    
    return K, top_left, top_right, bottom_left, bottom_right

def test_assemble_suboptimal_K(student_sol: callable, master_sol: callable, shouldprint: bool = True) -> bool:
    """
    Testing student solution for function ``assemble_suboptimal_K``.

    Parameters:
    - ``student_sol`` (callable): The function defined by the student
    - ``master_sol`` (callable): The function defined by the instructor

    Returns:
    - bool: The result of the unit test
    """
    A=np.array([[-10, 12, -1], [-1, 1.25, -2], [1, 0, -5]])
    B_w=np.array([[0, 1, -4], [-1, 0, 2], [0, -1, 1]])
    B_u=np.eye(3)
    C_z=np.array([[1, 0, -1], [1, 0, 8], [0, 6, 1]])
    C_y=np.eye(3)
    D_zw=np.zeros((3,3))
    D_zu=np.array([[-1, 0, -1], [2, 1, -3], [1, 10, 3]])
    D_yw=np.array([[1, 0, 3], [6, 1, -0.5], [-1, 0, 1]])
    D_yu=np.zeros((3,3))

    R_ww = D_yw @ D_yw.T
    gamma = 0.5

    try:
        student = student_sol(A, B_w, B_u, C_z, C_y, D_zw, D_zu, D_yw, D_yu, gamma)
    except Exception as e:
        print(f"Could not run student solution: {e}")
        return False
    master = master_sol(A, B_w, B_u, C_z, C_y, D_zw, D_zu, D_yw, D_yu, gamma)
    if np.allclose(student[0], master[0]):
        print("Student implementation of assemble_suboptimal_K passed the test")
        return True
    else:
        if shouldprint:
            print("Student implementation of assemble_suboptimal_K failed the test. Inspect ``assemble_suboptimal_K`` to see the solution.")
            print(f"Student solution K = : {student[0]}")
            print(f"Master solution K =: {master[0]}")
        return False