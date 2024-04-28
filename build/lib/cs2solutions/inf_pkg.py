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
    
