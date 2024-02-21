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


# Visualize the Noise signal in Morse-Transmission
def plot_noise_signal(t_cont: np.array, rand_noise: np.array,
            t_cont_start: int, t_cont_end: int) -> None:
  """
  Plot the continuous noise signal.
  
  Parameters:
   - ``t_cont`` (np.array): Array of time values for the continuous signal.
   - ``rand_noise`` (np.array): Array of amplitude values for the continuous noise signal.
   - ``t_cont_start`` (int): Start time for the x-axis.
   - ``t_cont_end`` (int): End time for the x-axis.

  Returns:
   - None
  """
  
  plt.figure(figsize=(5, 3))
  plt.title("(Continuous) Noise")
  plt.axhline(y=0, color='g', linestyle='-')
  plt.plot(t_cont, rand_noise)
  plt.xticks(np.arange(t_cont_start, t_cont_end))
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.show()
    
  return None


# Define the morse code dictionary.
MORSE_CODE_DICT = { 'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-','':''}

# Function to encrypt the string according to the Morse code chart.
def encrypt(message: str) -> str:
    """
    Encrypts a given message using Morse code.

    Parameters:
    - ``message`` (str): The message to be encrypted.

    Returns:
    - ``cipher`` (str): The encrypted message in Morse code.
    """
    cipher = ''
    for letter in message:
        if letter != ' ':
            cipher += MORSE_CODE_DICT[letter] + ' '
        else:
            cipher += ' '

    return cipher

# Function to decrypt the string from Morse to English.
def decrypt(message: str) -> str:
    """
    Decrypts a message encoded in Morse code.
    
    Parameters:
     - ``message`` (str): The Morse code message to be decrypted.
        
    Returns:
     - ``decipher`` (str): The decrypted message in English.
    """
    
    # Extra space added at the end to access the last morse code.
    message += ' '
    
    # Variables to keep track of the Morse code and the English translation.
    decipher = ''
    citext = ''
    for letter in message:
 
        # Checks for space.
        if (letter != ' '):
            # Counter to keep track of space.
            i = 0
 
            # Storing Morse code of a single character.
            citext += letter
 
        # In the case of space.
        else:
            # If i = 1 that indicates a new character.
            i += 1
 
            # If i = 2 that indicates a new word.
            if i == 2 :
                 # Adding space to separate words.
                decipher += ' '
            else:
                try:
                    # Accessing the keys using their values (reverse of encryption).
                    decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT.values()).index(citext)]
                    citext = ''
                except:
                    pass
 
    return decipher


def code_to_signal(t_cont: np.array,                
                   morse_code: str,
                   morse_com_freq: float) -> np.array:
  """
  Standardized signal to represent the Morse Signal:

  ".": 1 ON & 1 OFF
  "-": 2 ON & 1 OFF
  " ": 2 OFF

  This Function that turns these into ON and OFF signals.

  Parameters: 
  - ``t_cont`` (np.array): Array of time values for the continuous signal.
  - ``morse_code`` (str): The Morse code message to be transmitted.
  - ``morse_com_freq`` (float): The frequency of the Morse code signal.
      
  Output: 
  - ``morse_signal`` (np.array): The Morse code signal.
  """
  
  # A calculation turns the frequency to the period in [s].
  # The period defines how long a "." is (represented in signal as on).
  period = 1/morse_com_freq
  
  # Generate an empty array matching the size of the continuous time array for signal data storage.
  morse_signal = np.zeros(t_cont.size)

  e = 0
  i = 0

  # Check if counter within both array.
  while ((e < len(morse_code)) and (i < t_cont.size - 1)):
    # If yes, read the current character in Morse code and determine what to write on the signal data storage.
    
    """
    ".": 1 ON & 1 OFF
    "-": 2 ON & 1 OFF
    " ": 2 OFF
    """
    
    if morse_code[e] == '.':

      on_end = t_cont[i] + period
      off_end = t_cont[i] + 2*period

    if morse_code[e] == "-":
      on_end = t_cont[i] + 2*period
      off_end = t_cont[i] + 3*period

    if morse_code[e] == " ":
      on_end = t_cont[i]
      off_end = t_cont[i] + 2*period

    # After determining how long and what to write, write the signal on the data array.
    while ((t_cont[i] < on_end) and (i < t_cont.size - 1)):
      # Write the ON state.
      morse_signal[i] = 1
      i = i + 1

    while ((t_cont[i] <= off_end) and (i < t_cont.size - 1)):
      # Write the OFF state.
      morse_signal[i] = 0
      i = i + 1

    e = e + 1
    
  return morse_signal


def signal_to_code(t_dis: np.array,
                   dis_morse_signal: np.array,
                   morse_com_freq: float) -> str:
  """
  This function decodes signals back to Morse Code.

  Parameters:
  - ``t_dis`` (np.array): Discrete-Time Time Array.
  - ``dis_morse_signal`` (np.array): Discrete-Time Signal Value Array. Signal must be normalized to 1 and 0.
  - ``morse_com_freq`` (float): Morse Communication Frequency.

  Returns:
  - ``morse_code`` (str): Morse Code consisting of ".", "-", and " ".
  """

  # A calculation turns the frequency to the period in [s].
  # The period defines how long a "." is (represented in signal as on).
  period = 1/morse_com_freq
  
  i = 0
  p = 0
  state = 0
  hold_time = 0
  code = ""
  
  while i < len(t_dis)-1:
    
    while state == 0:
      # Check if the element index is still in the range of the data array.
      # Check if the state remains unchanged, hence signal is smaller than 0.8.
      # Check if the OFF is being held for too long, might mean that the message has ended.
      while i < len(t_dis)-1 and dis_morse_signal[i] < 0.8 and hold_time < 20*period:
        # Move to the next element.
        i += 1
        hold_time = t_dis[i] - t_dis[p]
        
      if i >= len(t_dis)-1:
        # This condition indicates wether the index is over the total elements, in which case the function should be terminated.
        return code
      
      if hold_time >= 10*period:
        # This condition indicates the message ends.
        code += "  "
        return code
      
      # Otherwise it indicates a state change from OFF to ON, and the message continues.
      if hold_time >= 0.6*period:
        if hold_time >= 2.5*period:
          if hold_time >= 4.2*period:
            code += "  "
          else:
            code += " "
      # Otherwise it was just a break to start another character.
      
      # Change the state to ON.
      state = 1
      # Refresh the timer.
      p = i
      
    while state == 1:
      # Check if the element index is still in the range of the data array.
      # Check if the state remains unchanged, hence signal is higher than 0.8.
      # Check if the ON is being held for too long, might mean that the message has an error.
      while i < len(t_dis)-1 and dis_morse_signal[i] >= 0.8 and hold_time < 20*period:
        # Move to the next element.
        i += 1
        hold_time = t_dis[i] - t_dis[p]
      
      if i >= len(t_dis)-1:
        # This condition indicates the index is over the total elements, in which case the function should be terminated.
        return code
      
      if hold_time >= 20*period:
        # This condition indicates the message ends.
        code += "  "
        return code
      
      # Otherwise it indicates a state change from ON to OFF, and the message continues.
      if hold_time >= 0.6*period:
        # If ON time is longer than 1.8, it's a "-".
        if hold_time >= 1.6*period:
          code += "-"
        else:
          code += "."
      
      # Change the state to OFF.
      state = 0
      # Refresh the timer.
      p = i
    
  return code


def butter_lowpass_filter(data: np.array,
                          cutoff_frequency: float,
                          sampling_frequency: float,
                          order: int) -> np.array:
    """
    Apply a low-pass Butterworth filter to the input signal data.
    
    Parameters:
    - ``data`` (np.array): 1D array of signal values.
    - ``cutoff_frequency`` (float): Cutoff frequency of the filter.
    - ``sampling_frequency`` (float): Sampling frequency of the data.
    - ``order`` (int): Order of the low-pass transfer function.
    
    Returns:
    - ``filtered_data`` (np.array): 1D array of filtered signal values.
    """
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the signal.
    filtered_data = lfilter(b, a, data)
    return filtered_data


def analog_digital_converter(t_cont: np.array,
                             cont_signal: np.array,
                             freq: float,
                             t_or_s: str) -> np.array:
    """
    This function selectively picks the data points from the continuous data array to simulate signal sampling.
    
    Parameters:
    - ``t_cont`` (np.array): Continues-Time Time Array.
    - ``cont_signal`` (np.array): Continues-Time Signal Value Array.
    - ``freq`` (float): ADC Sampling Frequency.
    - ``t_or_s`` (str): Array generator selector. 't' to generate a Discrete-Time Time Array, 's' to generate a Discrete-Time Signal Value Array.
    
    Returns:
    - ``t_dis`` OR ``dis_signal`` (np.array): Discrete-Time Time Array or Discrete-Time Signal Value Array, depending on the value of t_or_s.
    """
    
    # Converting the Sampling frequency to the sampling period.
    period = 1/freq
 
    # Initialize an empty array to store the time data for discrete time.
    t_dis = np.zeros(t_cont.size)
    dis_signal = np.zeros(t_cont.size)
 
    i = 0
    p = 0
    e = 0
 
    # Check if the element index is still in the range of the time array.
    while i < len(t_cont)-1:
        
        # Collect data from continuous time and write it to the sensor data array.
        t_dis[e] = t_cont[i]
        dis_signal[e] = cont_signal[i]
        
        # Move to the next element in the sensor data.
        e += 1
        
        # Keep skipping the time till the next time to be sampled by the sensor.
        # While checking if the element is still in range.
        while (t_cont[i] < t_cont[p] + period) and (i < len(t_cont)-1):
            i += 1
        # We skipped a period in continuous time, now it is time to collect data.
        p = i
    
    # If dis_time wanted, return dis_time array.
    if t_or_s == "t":
        return t_dis   
    
    # If dis_signal wanted, return dis_signal array
    if t_or_s == "s":
        return dis_signal
 
    return None
    

def data_normalization_sol(data_raw: np.array,
                           min: float,
                           normalization: float,
                           idle: float) -> np.array:
    """
    Normalize the input data array to discrete values of 1 or 0.
    If the signal value is higher than or equal to the minimum threshold, it is normalized to the normalization value.
    If the signal value is lower than the minimum threshold, it is normalized to the idle value.
    
    Parameters:
    - ``data_raw`` (np.array): 1D array of raw data points.
    - ``min`` (float): Minimum threshold for normalization.
    - ``normalization`` (float): Value to normalize to when the signal value is higher than or equal to the minimum threshold.
    - ``idle`` (float): Value to normalize to when the signal value is lower than the minimum threshold.
    
    Returns:
    - ``data_norm`` (np.array): 1D array of filtered data points with the same size as the raw data array.
    """
    # Create an array to save the filtered data with the same size as the raw data array.
    data_norm = np.zeros(data_raw.size)
    
    # Check each value and normalize or set it to the idle value in the filtered data array.
    for i in range(data_raw.size):
        if min <= data_raw[i]:
            data_norm[i] = normalization
        else:
            data_norm[i] = idle
    return data_norm