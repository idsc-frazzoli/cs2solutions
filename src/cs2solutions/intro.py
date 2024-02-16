
try:
    import control as ct
    
except ImportError:
    print('Error: Could not import control library. Please install it using the command "!pip install control"')

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Error: Could not import matplotlib library. Please install it using the command "!pip install matplotlib"')

try:
    import numpy as np
except ImportError:
    print('Error: Could not import numpy library. Please install it using the command "!pip install numpy"')

try:
    import unittest as unit
except ImportError:
    print('Error: Could not import unittest library. Please install it using the command "!pip install unittest"')

#solution function definitions

def sol_create_transfer_function() -> ct.TransferFunction:
    """
    Solution: Create a TransferFunction object to represent the transfer function s/(s^2+2*s+1).

    Returns:
    TransferFunction: The created TransferFunction object.
    """
    # Define system parameters
    numerator_coeffs = [1,0]
    denominator_coeffs = [1, 2, 1]

    # Create a transfer function
    tf = ct.TransferFunction(numerator_coeffs, denominator_coeffs)
    return tf

def test_create_transfer_function(student_sol, sol_tf, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's transfer function with the solution transfer function.

    Parameters:
    - student_sol: A function that returns the student's transfer function.
    - sol: A function that returns the solution transfer function.
    - shouldprint: A boolean indicating whether to print the transfer functions. Default is True.

    Raises:
        AssertionError: If the input is not an instance of the TransferFunction class.

    Returns:
    - A boolean indicating whether the student's transfer function is equal to the solution transfer function.
    """
    # Create a transfer function using the student's code
    student_tf = student_sol()
    # Create a transfer function using the solution code
    sol_tf = sol_tf()

    # Check if the student's transfer function is an instance of the TransferFunction class
    assert isinstance(student_tf, ct.TransferFunction), "The function should return an instance of the TransferFunction class."
    assert isinstance(sol_tf, ct.TransferFunction), "The function should return an instance of the TransferFunction class."

    # Check if the student's transfer function is equal to the solution transfer function
    if (shouldprint):
        print("Student's transfer function: ", student_tf)
        print("Solution transfer function: ", sol_tf)

    return np.array_equal(student_tf.num, sol_tf.num) and np.array_equal(student_tf.den, sol_tf.den)


def sol_plot_step_response(tf: ct.TransferFunction):
    """
    Solution: Plot the step response of the given transfer function.

    Parameters:
    tf (TransferFunction): The transfer function to plot the step response for.

    Raises:
        AssertionError: If the input is not an instance of the TransferFunction class.

    Returns:
    None
    """
    assert isinstance(tf, ct.TransferFunction), "The input should be an instance of the TransferFunction class."

    # Compute the step response of the system
    time, response = ct.step_response(tf)

    # Plot the step response
    plt.figure()
    plt.plot(time, response)
    plt.title('Step Response of the System')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def test_plot_step_response(student_plot, sol_plot, tf: ct.TransferFunction, shouldprint: bool = True) -> bool:
    """
    Test function to compare the student's step response plot with the solution step response plot.

    Parameters:
    - student_plot: A function that plots the student's step response.
    - sol_plot: A function that plots the solution step response.
    - shouldprint: A boolean indicating whether to print the plots. Default is True.

    Raises:
        AssertionError: If the input is not an instance of the TransferFunction class.

    Returns:
    - A boolean indicating whether the student's step response plot is equal to the solution step response plot.
    """
    assert isinstance(tf, ct.TransferFunction), "The input should be an instance of the TransferFunction class."

    # Plot the step response using the student's code
    print("Student's plot:") if shouldprint else None
    student_plot(tf)
    # Plot the step response using the solution code
    print("Solution plot:") if shouldprint else None
    sol_plot(sol_create_transfer_function())

    pass


def sol_bode_plot(tf: ct.TransferFunction):
    """
    Solution: Plot the Bode magnitude and phase plots for a given transfer function.

    Parameters:
        tf (ct.TransferFunction): The transfer function to plot.

    Raises:
        AssertionError: If the input is not an instance of the TransferFunction class.
    """
    assert isinstance(tf, ct.TransferFunction), "The input should be an instance of the TransferFunction class."

    mag, phase, omega = ct.bode_plot(tf)
    pass


def sol_get_filter_tf(omega_c: float, is_verbose : bool = False) -> ct.TransferFunction:
    """
    Creates a transfer function for a low pass filter.

    Parameters:
    omega_c (float): The cutoff frequency of the filter.
    is_verbose (bool, optional): If True, prints additional information about the filter. Default is False.

    Returns:
    ct.TransferFunction: The transfer function of the low pass filter.
    """
    assert isinstance(omega_c, float) or isinstance(omega_c, int), "The cutoff frequency should be a number."
    assert omega_c > 0, "The cutoff frequency should be greater than 0."

    if is_verbose:
        print("It can be show that the parameter \"a\" from Exercise 4.1 is equal to (omega_c)^-1.\nTherefore, the TF of the low pass filter is:")

    # Define system parameters
    numerator_coeffs = [1.0]
    denominator_coeffs = [1.0 / omega_c, 1.0]

    # Create a transfer function
    tf = ct.TransferFunction(numerator_coeffs, denominator_coeffs)
    return tf

def test_create_filter_tf(student_filter, sol_filter, omega_c: float, shouldprint: bool = True) -> bool:
    """
    Test the create_filter_tf function.

    Parameters:
    student_filter (function): The student's implementation of the filter function.
    sol_filter (function): The solution's implementation of the filter function.
    omega_c (float): The cutoff frequency.
    shouldprint (bool, optional): Whether to print the test results. Defaults to True.

    Returns:
    bool: True if the student's implementation matches the solution's implementation, False otherwise.
    """
    assert isinstance(omega_c, float) or isinstance(omega_c, int), "The cutoff frequency should be a number."
    assert isinstance(student_filter(omega_c), ct.TransferFunction), "The function should return an instance of the TransferFunction class."
    assert isinstance(sol_filter(omega_c), ct.TransferFunction), "The function should return an instance of the TransferFunction class."

    student_tf = student_filter(omega_c)
    sol_tf = sol_filter(omega_c)

    return np.array_equal(student_tf.num, sol_tf.num) and np.array_equal(student_tf.den, sol_tf.den)

#test function definitions

def test(answer, solution, is_verbose: bool = True):
    #this is a very simple test to check correct inputs and plotting
    if(is_verbose):
        print(f"Your answer: {answer()}")
        print(f"Expected answer: {solution()}")
    else:
        print("Your answer: ")
        answer()
        print("Expected answer: ")
        solution()
    pass
       