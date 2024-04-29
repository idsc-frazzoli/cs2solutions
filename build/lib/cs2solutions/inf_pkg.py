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

def SISO(system_number: int) -> ct.TransferFunction:
    """
    This function creates a SISO transfer function object

    Args:
    - ``num`` (int): The numerator of the transfer function

    Returns:
    - ``ct.TransferFunction``: The transfer function object
    """
    systems = {
        0: ([0, 0, 1], [1, 2, 4]),
        1: ([0, 0, 1], [0, 1, 1]),
        2: ([0, 0, 1], [0, 1, -1]),
        3: ([0, 1, 1], [1, 4, 9]),
        4: ([0, 5, 5], [0, 1, 1]),
        5: ([0, 11, 11], [0, 1, -1]),
        6: ([2, 0, -2], [1, 4, 9]),
    }

    num, den = systems[system_number]
    return ct.TransferFunction(num, den)
    
def sol_systemInfnorm(system: ct.TransferFunction) -> float:
    """
    Returns an approximation of the infinity norm of the system

    Parameters:
    - ``system`` (ct.TransferFunction): The system to compute the infinity norm of

    Returns:
    - float: The infinity norm of the system
    """
    # Create a range of frequencies to analyze over
    omega = np.linspace(-4, 4, 1000)
    H = system(omega * 1j)

    # Consider the MIMO case
    if system.ninputs > 1 or system.noutputs > 1:
        # Calculate singular values
        singular_values = [np.linalg.svd(H[..., i])[1] for i in range(len(omega))]
    # Consider the SISO case
    else:
        singular_values = [np.absolute(H[..., i]) for i in range(len(omega))]

    # Return the highest singular value
    return np.vstack(singular_values).max()

def sol_is_stable(system: ct.TransferFunction) -> bool:
    """
    Returns whether the system is stable

    Parameters:
    - ``system`` (ct.TransferFunction): The system to check for stability

    Returns:
    - bool: Whether the system is stable
    """
    return all(np.real(system.poles) < 0)

def sol_small_gain_theorem(systems: list[ct.TransferFunction]) -> bool:
    """
    Checks if the small gain theorem is satisfied for the given systems.

    Parameters:
    - ``systems`` (list[ct.TransferFunction]): The systems to check the small gain theorem for

    Returns:
    - bool: Whether the small gain theorem is satisfied
    """
    if not all(sol_is_stable(system) for system in systems):
        return False
    
    list_gamma = [sol_systemInfnorm(system) for system in systems]
    return np.prod(list_gamma) < 1

def sol_internal_stability_check(toptf: ct.TransferFunction, bottomtf: ct.TransferFunction) -> bool:
    """
    Checks if the internal stability condition is satisfied for the given systems.

    Parameters:
    - ``toptf`` (ct.TransferFunction): The top transfer function
    - ``bottomtf`` (ct.TransferFunction): The bottom transfer function
    
    Returns:
    - bool: Whether the internal stability condition is satisfied
    """
    # First, create the overall transfer function
    return False