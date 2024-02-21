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

# Plotting Functions
        
def plot_helpers() -> None:
  """
  This function sets up the plot with appropriate labels and legend.
  
  Parameters: 
   - None

  Returns: 
   - None
  """
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.legend()


def plot_step_responses(t_stable_cont: np.array, y_stable_cont: np.array,
                        t_stable_eulerf: np.array, y_stable_eulerf: np.array,
                        t_stable_eulerb: np.array, y_stable_eulerb: np.array,
                        t_stable_tust: np.array, y_stable_tust: np.array,
                        t_stable_zoh: np.array, y_stable_zoh: np.array) -> None:
    """
    This function plots the step responses of a systems using different discretization methods. Plotted: Continuous, Forward Euler, Backward Euler, Tustin, and Zero Order Hold.

    Parameters:
    - ``t_stable_cont`` (np.array): Time array for the continuous system.
    - ``y_stable_cont`` (np.array): Output array for the continuous system.
    - ``t_stable_eulerf`` (np.array): Time array for the Forward Euler system.
    - ``y_stable_eulerf`` (np.array): Output array for the Forward Euler system.
    - ``t_stable_eulerb`` (np.array): Time array for the Backward Euler system.
    - ``y_stable_eulerb`` (np.array): Output array for the Backward Euler system.
    - ``t_stable_tust`` (np.array): Time array for the Tustin system.
    - ``y_stable_tust`` (np.array): Output array for the Tustin system.
    - ``t_stable_zoh`` (np.array): Time array for the Zero Order Hold system.
    - ``y_stable_zoh`` (np.array): Output array for the Zero Order Hold system.

    Returns:
    - None
    """
    
    plt.figure(figsize=[24, 16])
    plt.subplot(3, 2, 1)
    plt.plot(t_stable_cont, y_stable_cont, label='Continuous System')
    plt.step(t_stable_eulerf, y_stable_eulerf, label='Forward Euler')
    plt.step(t_stable_eulerb, y_stable_eulerb, label='Backward Euler')
    plt.step(t_stable_tust, y_stable_tust, label='Tustin')
    plt.step(t_stable_zoh, y_stable_zoh, label='Zero Order Hold')

    # Adding labels and title to the plot.
    
    plt.title('Step Response Combined')
    plot_helpers()
    

    plt.subplot(3, 2, 2)
    plt.plot(t_stable_cont, y_stable_cont, label='Continuous System', color = 'C0')

    # Adding labels and title to the plot.
    plt.title('Step Response Continuous')
    plot_helpers()


    plt.subplot(3, 2, 3)
    plt.step(t_stable_eulerf, y_stable_eulerf, label='Forward Euler', color = 'C1')

    # Adding labels and title to the plot.
    plt.title('Step Response Forward Euler')
    plot_helpers()


    plt.subplot(3, 2, 4)
    plt.step(t_stable_eulerb, y_stable_eulerb, label='Backward Euler', color = 'C2')

    # Adding labels and title to the plot.
    plt.title('Step Response Backward Euler')
    plot_helpers()


    plt.subplot(3, 2, 5)
    plt.step(t_stable_tust, y_stable_tust, label='Tustin', color = 'C3')

    # Adding labels and title to the plot.
    plt.title('Step Response Tustin')
    plot_helpers()


    plt.subplot(3, 2, 6)
    plt.step(t_stable_zoh, y_stable_zoh, label='Zero Order Hold', color = 'C4')

    # Adding labels and title to the plot.
    plt.title('Step Response Zero Order Hold')
    plot_helpers()

    return None
    

def plot_disc_stepresponses(t_cont: np.array, y_cont: np.array,
              t_eulerf: np.array, y_eulerf: np.array,
              t_eulerb: np.array, y_eulerb: np.array,
              t_tust: np.array, y_tust: np.array,
              t_zoh: np.array, y_zoh: np.array, Ts: float) -> None:
  """
  Plots the step responses of a discrete system along with the continuous system.
  
  Parameters:
   - ``t_cont`` (np.array): Time array for the continuous system.
   - ``y_cont`` (np.array): Output array for the continuous system.
   - ``t_eulerf`` (np.array): Time array for the forward Euler method.
   - ``y_eulerf`` (np.array): Output array for the forward Euler method.
   - ``t_eulerb`` (np.array): Time array for the backward Euler method.
   - ``y_eulerb`` (np.array): Output array for the backward Euler method.
   - ``t_tust`` (np.array): Time array for the Tustin method.
   - ``y_tust`` (np.array): Output array for the Tustin method.
   - ``t_zoh`` (np.array): Time array for the Zero Order Hold method.
   - ``y_zoh`` (np.array): Output array for the Zero Order Hold method.
   - ``Ts`` (float): Sampling time of the discrete system.
  
  Returns:
   - None
  """
  plt.figure(figsize=(10, 6))
  
  # Inserting initial values for step response plotting
  t_eulerf = np.insert(np.squeeze(t_eulerf)+Ts,0,0)
  y_eulerf = np.insert(np.squeeze(y_eulerf), 0, 0)
  t_eulerb = np.insert(np.squeeze(t_eulerb)+Ts,0,0)
  y_eulerb = np.insert(np.squeeze(y_eulerb), 0, 0)
  t_tust = np.insert(np.squeeze(t_tust)+Ts,0,0)
  y_tust = np.insert(np.squeeze(y_tust), 0, 0)
  t_zoh = np.insert(np.squeeze(t_zoh)+Ts,0,0)
  y_zoh = np.insert(np.squeeze(y_zoh), 0, 0)
  
  # Plotting the step responses
  plt.ylim(-50, 50)
  plt.plot(t_cont, y_cont, label='Continuous System')
  plt.step(t_eulerf, y_eulerf, label='Forward Euler')
  plt.step(t_eulerb, y_eulerb, label='Backward Euler')
  plt.step(t_tust, y_tust, label='Tustin')
  plt.step(t_zoh, y_zoh, label='Zero Order Hold')
  plt.title('Discrete Step Response')
  plt.xlabel('Time (s)')
  plt.ylabel('Output')
  plt.legend()
  plt.show()

  return None
        

# Exercise 1a: Solutions

def cont_solution (A: np.array, B: np.array, C:np.array, D:np.array) -> Tuple[np.array, np.array]:
    """
    Calculates the step response of a continuous-time linear time-invariant system.
    
    Parameters:
     - ``A`` (np.array): State matrix of the system.
     - ``B`` (np.array): Input matrix of the system.
     - ``C`` (np.array): Output matrix of the system.
     - ``D`` (np.array): Direct transmission matrix of the system.
    
    Returns:
     - Tuple[np.array, np.array]: A tuple containing the time values and the corresponding output values of the step response.
    """
    
    sysc = sig.lti(A, B, C, D)

    t_cont, y_cont = sysc.step(T = np.linspace(0, 25, 1000))
    
    return [t_cont, y_cont]

def disc_solution(A: np.array, B: np.array, C: np.array, D: np.array, Ts: float) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Calculates the discrete-time step response using different methods.

    Parameters:
     - ``A`` (np.array): State matrix of the continuous-time system.
     - ``B`` (np.array): Input matrix of the continuous-time system.
     - ``C`` (np.array): Output matrix of the continuous-time system.
     - ``D`` (np.array): Feedthrough matrix of the continuous-time system.
     - ``Ts`` (float): Sampling time.

    Returns:
    Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]: A tuple containing the time and output arrays for each method:
        - ``t_eulerf``: Time array for the forward Euler method.
        - ``y_eulerf``: Output array for the forward Euler method.
        - ``t_eulerb``: Time array for the backward difference method.
        - ``y_eulerb``: Output array for the backward difference method.
        - ``t_tust``: Time array for the Tustin method.
        - ``y_tust``: Output array for the Tustin method.
        - ``t_zoh``: Time array for the zero-order hold method.
        - ``y_zoh``: Output array for the zero-order hold method.
    """
        
    sysd_eulerf = sig.cont2discrete((A, B, C, D), Ts, method='euler')
    t_eulerf, y_eulerf = sig.dstep(sysd_eulerf, t=np.arange(0, 25, Ts))

    sysd_eulerb = sig.cont2discrete((A, B, C, D), Ts, method='backward_diff')
    t_eulerb, y_eulerb = sig.dstep(sysd_eulerb, t=np.arange(0, 25, Ts))

    sysd_tust = sig.cont2discrete((A, B, C, D), Ts, method='bilinear')
    t_tust, y_tust = sig.dstep(sysd_tust, t=np.arange(0, 25, Ts))

    sysd_zoh = sig.cont2discrete((A, B, C, D), Ts, method='zoh')
    t_zoh, y_zoh = sig.dstep(sysd_zoh, t=np.arange(0, 25, Ts))
    
    return [t_eulerf, y_eulerf, t_eulerb, y_eulerb, t_tust, y_tust, t_zoh, y_zoh]


# Ackermann's Formula

def feedforward_kr(A: np.array, B: np.array, C: np.array, K: np.array) -> np.array:
    """
    Calculates the feedforward gain (kr) for a given system.

    Parameters:
     - ``A`` (np.array): The state matrix of the system.
     - ``B`` (np.array): The input matrix of the system.
     - ``C`` (np.array): The output matrix of the system.
     - ``K`` (np.array): The feedback gain matrix of the system.

    Returns:
     - ``kr`` (np.array): The feedforward gain (kr) of the system.
    """

    ABK = A - B @ K
    ABK_inv = np.linalg.inv(ABK)
    den = C @ ABK_inv @ B
    kr = - 1/den

    return kr


def acker(A: np.array, B: np.array, poles: List[float]) -> np.array:
    """
    Calculates the state feedback gain matrix K using the Ackermann's formula.

    Parameters:
     - ``A`` (np.array): State matrix of the system.
     - ``B`` (np.array): Input matrix of the system.
     - ``poles`` (List[float]): List of desired closed-loop poles.

    Returns:
     - ``K`` (np.array): State feedback gain matrix K.
    """

    AB = A @ B
    R = np.concatenate((B, AB), axis=1)
    R_inv = np.linalg.inv(R)
    gamma = np.array([[0, 1]]) @ R_inv

    p_1 = poles[0]*(-1)
    p_2 = poles[1]*(-1)
    ab = p_1 + p_2
    b = p_1*p_2
    p_cl = A @ A + ab*A + b*np.identity(2)

    K = gamma @ p_cl

    return K


def place_poles_Ackermann(A: np.array, B: np.array, C: np.array,
                          D: float, poles: List) -> np.array:
    """
    Computes the controller gains and closed loop system dynamics using the Ackermann method
    to place the poles of the system at desired locations.
    
    Parameters:
     - ``A`` (np.array): State matrix of the system.
     - ``B`` (np.array): Input matrix of the system.
     - ``C`` (np.array): Output matrix of the system.
     - ``D`` (float): Direct transmission matrix of the system.
     - ``poles`` (List): List of desired pole locations.
        
    Returns:
     - ``clsys`` (np.array): Updated system dynamics using new poles and feedforward gain.
    """
    
    K=ct.acker(A, B, poles)
    
    # Create a new system representing the closed loop response.
    clsys = ct.StateSpace(A - B @ K, B, C, D)

    # Compute the feedforward gain.
    kr = feedforward_kr(A, B, C, K)
    
    # Scale the input by the feedforward gain.
    clsys *= kr

    # Return gains and closed loop system dynamics.
    return clsys


# CURRENTLY UNABLE TO UPDATE ACKERMANN!


# Discretization Functions

def euler_forward(system, dt) -> ct.StateSpace:
    """
    Discretize a continuous-time system using the Euler forward method.

    Parameters:
     - ``system`` (ct.StateSpace): State-space system (instance of ct.StateSpace).
     - ``dt`` (float): Time step for discretization.

    Returns:
     - StateSpace(Ad, Bd, Cd, Dd): Discretized state-space matrices.
    """
    A, B, C, D = system.A, system.B, system.C, system.D
    I = np.eye(A.shape[0])
    Ad = I + dt * A
    Bd = dt * B
    Cd = C
    Dd = D
    return ct.StateSpace(Ad, Bd, Cd, Dd)


def euler_backward(system, dt) -> ct.StateSpace:
    """
    Discretize a continuous-time system using the Euler backward method.

    Parameters:
     - ``system`` (ct.StateSpace): State-space system (instance of ct.StateSpace).
     - ``dt`` (float): Time step for discretization.

    Returns:
     -   StateSpace(Ad, Bd, Cd, Dd): Discretized state-space matrices.
    """
    A, B, C, D = system.A, system.B, system.C, system.D
    I = np.eye(A.shape[0])
    Ad = np.linalg.inv(I - dt * A)
    Bd = np.dot(Ad, dt * B)
    Cd = C
    Dd = D
    return ct.StateSpace(Ad, Bd, Cd, Dd)


def tustin_method(system, dt) -> ct.StateSpace:
    """
    Discretize a continuous-time system using Tustin's method (bilinear transformation).
    
    Parameters:
     - ``system`` (ct.StateSpace): State-space system (instance of ct.StateSpace).
     - ``dt`` (float): Time step for discretization.
    
    Returns:
     - StateSpace(Ad, Bd, Cd, Dd): Discretized state-space matrices.
    """
    A, B, C, D = system.A, system.B, system.C, system.D
    I = np.eye(A.shape[0])
    pre_matrix = np.linalg.inv(I - (dt / 2) * A)
    Ad = np.dot(pre_matrix, I + (dt / 2) * A)
    Bd = np.dot(pre_matrix, dt * B)
    Cd = C
    Dd = D
    return ct.StateSpace(Ad, Bd, Cd, Dd)


def discretize(system: ct.StateSpace, dt: float) -> Tuple[ct.StateSpace, ct.StateSpace, ct.StateSpace]:
    """
    Discretizes a continuous-time system into discrete-time systems using three methods: 
    Euler Forward, Euler Backward, and Tustin's method.

    Parameters:
     - ``system`` (ct.StateSpace): A continuous-time system represented as a ct.StateSpace object.
     - ``dt`` (float): Time step for discretization.

    Returns:
     - Tuple[ct.StateSpace, ct.StateSpace, ct.StateSpace]: A tuple of discretized systems (dt_forward, dt_backward, dt_tustin), each representing 
              the input system discretized using Euler Forward, Euler Backward, and Tustin's method, respectively.
    """
    # Discretize using each method
    dt_forward = euler_forward(system, dt)
    dt_backward = euler_backward(system, dt)
    dt_tustin = tustin_method(system, dt)
    
    return dt_forward, dt_backward, dt_tustin


def plot_eigenvalues(system, method_name):
        """
        Plots the eigenvalues of the discretized system's A matrix on the complex plane.

        Parameters:
         - ``system`` (ct.StateSpace): A discrete-time system represented as a ct.StateSpace object, 
                                                          whose eigenvalues are to be plotted.
         - ``method_name`` (str): The name of the discretization method used for this system. 
                                                  This name is used as a label in the plot.

        Returns:
         - None

        The function calculates the eigenvalues of the system's A matrix, prints them,
        and plots them on the complex plane. Each set of eigenvalues is labeled according 
        to the discretization method used.
        """
        Ad = system.A
        eigenvalues = np.linalg.eigvals(Ad)
        print("Eigenvalues using " + method_name)
        print(eigenvalues)
        print("Magnitude")
        print(np.linalg.norm(eigenvalues[0]))
        print("\n")

        plt.scatter(eigenvalues.real, eigenvalues.imag, label=method_name)

        return None


def plot_table(dt_forward, dt_backward, dt_tustin):
    """
    Plots the eigenvalues of three discretized systems on the complex plane to compare their stability.

    Parameters:
     - ``dt_forward`` (ct.StateSpace): Discretized system using the Euler Forward method (StateSpace object).
     - ``dt_backward`` (ct.StateSpace): Discretized system using the Euler Backward method (StateSpace object).
     - ``dt_tustin`` (ct.StateSpace): Discretized system using Tustin's method (StateSpace object).

    This function visualizes the eigenvalues of the given discretized systems on the complex plane.
    It helps in comparing the stability characteristics of the systems discretized using different methods.
    The function plots each set of eigenvalues with a different label corresponding to the discretization method.
    It also includes a unit circle for reference, aiding in the assessment of stability (eigenvalues inside the 
    unit circle indicate stability in discrete-time systems).

    Returns:
     - None
    """
    plt.figure(figsize=(8, 6))
    
    plot_eigenvalues(dt_forward, 'Euler Forward')
    plot_eigenvalues(dt_backward, 'Euler Backward')
    plot_eigenvalues(dt_tustin, 'Tustin')
    
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Eigenvalues of Discretized Systems')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    plt.grid(True)

    # adding unit circle for visualization
    theta = np.linspace(0, 2 * np.pi, 100)  # 100 points between 0 and 2*pi
    x = np.cos(theta)  # x coordinates on the circle
    y = np.sin(theta)  # y coordinates on the circle
    plt.plot(x, y, linestyle='--', color='grey')

    plt.show()

    return None


def simulate_step_response(system, dt, label, n=100):
    """
    Simulates and plots the step response of a discretized system.

    Parameters:
     - ``system`` (ct.StateSpace): A StateSpace object representing the discretized system.
     - ``dt`` (float): The sampling time step used for the discretization.
     - ``label`` (str): A string label to identify the step response in the plot.
     - ``n`` (int): The number of time steps to simulate (default is 100).

    This function calculates the step response of a given discretized system using the provided time step (dt) over 'n' time steps. 
    It then plots the response using matplotlib, labeling the plot with the given label for easy identification. 
    The step response is visualized using a step plot, which shows how the system's output evolves over time in response to a unit step input.

    Returns:
     - None
    """
    lti = sig.dlti(system.A, system.B, system.C, system.D, dt=dt)
    t, y = sig.dstep(lti, n=n)
    plt.step(t, np.squeeze(y), label=label)

    return None


# Exercise 1b: Test Ackermann + Discretization

def test_ackermann_discrete(A: np.array, 
                              B: np.array, 
                              C: np.array,
                              D: np.array, 
                              p_des: Tuple[complex, complex] = [-0.5+3j, -0.5-3j], 
                              Ts: float = 0.25) -> None:
    """
    Test the user's implementations of the closed-loop system dynamics and step response calculations.

    This function computes the closed-loop system dynamics using Ackermann's formula and
    calculates the step response of both the continuous and discrete closed-loop systems
    using different discretization schemes. It then plots the step responses.

    Parameters:
     - ``A`` (np.array): numpy array representing the system matrix A.
     - ``B`` (np.array): numpy array representing the input matrix B.
     - ``C`` (np.array): numpy array representing the output matrix C.
     - ``D`` (np.array): float representing the direct transmission matrix D.
     - ``p_des`` (List[complex, complex]): List of complex numbers representing the desired poles of the closed-loop system. Default value is [-0.5+3j, -0.5-3j].
     - ``Ts`` (float): float representing the sampling time for the discrete closed-loop system. Default value is 0.25.

    Returns:
     - None
    """

    # Compute the closed loop system dynamics using Ackermann's formula.
    clsys_p = place_poles_Ackermann(A=A, B=B, C=C, D=0,
                                        poles=p_des)
    t_p, y_p = ct.step_response(clsys_p, input=0, output=0)

    # Define the system matrices and sampling time
    A_stable = clsys_p.A
    B_stable = clsys_p.B
    C_stable = clsys_p.C
    D_stable = clsys_p.D



    # Compute the step response of the continuous closed loop system. 
    # If you couldn't solve the previous exercise, you can use the solution. For that replace the function call with the following:
    # [t_stable_cont, y_stable_cont] = cont_solution(A_stable, B_stable, C_stable, D_stable)
    t_stable_cont, y_stable_cont = continuous_step_response(A_stable, B_stable, C_stable, D_stable)



    # Compute the step response of the discrete closed loop system using the different discretization schemes. 
    # If you couldn't solve the previous exercise, you can use the solution. For that replace the function call with the following:
    # [t_stable_eulerf, y_stable_eulerf, t_stable_eulerb, y_stable_eulerb, t_stable_tust, y_stable_tust, t_stable_zoh, y_stable_zoh] = disc_solution(A_stable, B_stable, C_stable, D_stable, Ts)
    t_stable_eulerf, y_stable_eulerf = discrete_step_response_euler_forward(A_stable, B_stable, C_stable, D_stable, Ts)
    t_stable_eulerb, y_stable_eulerb = discrete_step_response_euler_backward(A_stable, B_stable, C_stable, D_stable, Ts)
    t_stable_tust, y_stable_tust = discrete_step_response_tustin(A_stable, B_stable, C_stable, D_stable, Ts)
    t_stable_zoh, y_stable_zoh = discrete_step_response_zoh(A_stable, B_stable, C_stable, D_stable, Ts)



    # Package the results for plotting.
    values_to_plot = [t_stable_cont, y_stable_cont,
                    t_stable_eulerf, y_stable_eulerf,
                    t_stable_eulerb, y_stable_eulerb,
                    t_stable_tust, y_stable_tust,
                    t_stable_zoh, y_stable_zoh]

    # Plot the step responses.
    plot_step_responses(*values_to_plot)

    return None