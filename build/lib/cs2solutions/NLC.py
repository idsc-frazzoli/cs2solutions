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

m = 800 # kg
rho = 1.293 # kg/m^3
c_d = 1.1 # drag coefficient
A = 1.38 # m^2

gamma = 1/(2*m) * rho * c_d * A

def sol_nonlinear_sys(t: float, v: float, u: float, params) -> float:
    """
    Master solution for the function ``nonlinear_sys()``. This can be interpreted as the first equation of the state-space equation.

    Parameters:
    - ``v`` (float): The input value.
    - ``u`` (float): The input value.

    Returns:
    - ``float``: The output value.
    """
    vdot = -gamma*v**2 + u
    return vdot

def sol_nonlinear_sys_out(t: float, v: float, u: float, params) -> float:
    """
    Master solution for the function ``nonlinear_sys_out()``. This can be interpreted as the second equation of the state-space equation.

    Parameters:
    - ``v`` (float): The input value.

    Returns:
    - ``float``: The output value.
    """
    y = v
    return y

def sol_simulate_nonlinear(sol_nonlinear_sys: callable, sol_nonlinear_sys_out: callable, u: Optional[int], initial_v: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master solution for the function ``simulate_sys()``. This function simulates the nonlinear system.

    Parameters:
    - ``sol_nonlinear_sys`` (callable): The function that represents the first equation of the state-space equation.
    - ``u`` (float): The input value.
    - ``delta_v`` (float): A change in wind.

    Returns:
    - ``Tuple[np.ndarray, np.ndarray]``: A tuple containing the time and the output values.
    """
    if u is None: 
        u = 0
        print("No input acceleration u provided. Defaulting to u = 0 m/s^2")
    if initial_v is None:
        initial_v = 0

    sys = ct.NonlinearIOSystem(sol_nonlinear_sys, sol_nonlinear_sys_out)
    t = np.linspace(0, 30, 1000)

    t_out_1, y_out_1 = ct.input_output_response(sys, T=t, U=u, X0=[initial_v])

    return t_out_1, y_out_1

def sol_find_equilibrium(sys: ct.NonlinearIOSystem, u: int) -> Tuple[float, float, float]:
    """
    Master solution to find the stagnation speed of the racecar. This function finds the equilibrium point of the system.

    Parameters:
    - ``sys`` (ct.NonlinearIOSystem): The nonlinear system.
    - ``u`` (float): The input value.

    Returns:
    - ``None``
    """
    initial_guess = 0.01
    v_eq, u_eq, y_eq = ct.find_eqpt(sys, [initial_guess], [u,], return_y=True)

    print("Input acceleration u_eq =", u_eq[0], "m/s^2")
    print("Stagnation speed v_eq =", v_eq[0],"m/s")
    return v_eq[0], u_eq[0], y_eq[0]

def sol_linearize_system(sys: ct.NonlinearIOSystem, v_eq: float, u_eq: float) -> Tuple[ct.StateSpace, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Master solution to linearize the system. This function linearizes the system around the equilibrium point.

    Parameters:
    - ``sys`` (ct.NonlinearIOSystem): The nonlinear system.
    - ``v_eq`` (float): The equilibrium value of the state variable.
    - ``u_eq`` (float): The equilibrium value of the input variable.

    Returns:
    - ``Tuple[ct.StateSpace, np.ndarray, np.ndarray]``: A tuple containing the linearized system, the A matrix, and the B matrix.
    """
    linearized_sys = ct.linearize(sys,[v_eq],[u_eq])
    A_matrix = linearized_sys.A

    return linearized_sys, A_matrix, linearized_sys.B, linearized_sys.C, linearized_sys.D

def sol_simulate_linear(linearized_sys: ct.statesp.StateSpace, delta_u: Optional[int], delta_v: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function simulates the linearized system around the equilibrium point.

    Parameters:
    - ``linearized_sys`` (ct.statesp.StateSpace): The linearized system.
    - ``delta_u`` (float): Change in input (u) from equilibrium.
    - ``delta_v`` (float): Change in input (v) from equilibrium.

    Returns:
    - ``Tuple[np.ndarray, np.ndarray]``: A tuple containing the time and the output values.
    """
    if delta_u is None: 
        delta_u = 0
        print("No change in acceleration delta_u provided. Defaulting to u = 20 m/s^2. Keep in mind that this system is linearized about the equilibrium point u = 20 m/s^2.")
    if delta_v is None:
        delta_v = 0
        print("No change in wind delta_v provided. Defaulting to delta_v = 0 m/s")
    
    t = np.linspace(0, 30, 3000)
    t_out_2, delta_y_out_2 = ct.input_output_response(linearized_sys, T=t, U=delta_u, X0=delta_v)
    return t_out_2, delta_y_out_2

def sol_V(v: float, v_eq: float) -> float:
    return 0.5 * (v - v_eq)**2

# Compute the time derivative of the Lyapunov function
def sol_V_dot(v: float, gamma: float, u: float, y_eq: float) -> float:
    return (v - y_eq) * (-gamma * v**2 + u)