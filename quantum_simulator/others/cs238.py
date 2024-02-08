import numpy as np
from numpy import exp, sqrt, pi

def flip_at_index(binary_string, index):
    """
    Flip the character at the specified index in a binary string.
    
    :param binary_string: A string consisting of '0's and '1's.
    :param index: The index of the character to flip.
    :return: The modified string with the character at the given index flipped.
    """
    if index < 0 or index >= len(binary_string):
        raise IndexError("Index out of range")

    # Convert the string to a list
    char_list = list(binary_string)

    # Flip the character at the specified index
    char_list[index] = '0' if char_list[index] == '1' else '1'

    # Convert the list back to a string and return
    return ''.join(char_list)

def replace_at_index(binary_string, index):
    """
    Replace the character at the specified index in a binary string with '0' and '1'.

    :param binary_string: A string consisting of '0's and '1's.
    :param index: The index of the character to replace.
    :return: Two strings, one with the character at the index replaced by '0' and the other by '1'.
    """
    if index < 0 or index >= len(binary_string):
        raise IndexError("Index out of range")

    # Convert the string to a list
    char_list = list(binary_string)

    # Create two copies of the list
    list_with_0 = char_list.copy()
    list_with_1 = char_list.copy()

    # Replace the character at the specified index
    list_with_0[index] = '0'
    list_with_1[index] = '1'

    # Convert the lists back to strings
    return ''.join(list_with_0), ''.join(list_with_1)

def round_complex(z):
    """
    Round the real and imaginary parts of a complex number to five significant figures.

    :param z: Complex number to be rounded.
    :return: Complex number with real and imaginary parts rounded to five significant figures.
    """
    return complex(round(z.real, 5), round(z.imag, 5))

class QuantumSimulator:
    def __init__(self, N):
        """
        Initialize the attributes to describe a QuantumSimulator with number of qubits.
        """
        initial_state = '0' * N
        state_vector = {initial_state : 1}
        self.state = state_vector
        
    def x(self, target):
        new_state = {}
        for key, value in self.state.items():
            new_key, new_value = key, value
            new_key = flip_at_index(new_key, target)
            new_state[new_key] = new_value
        self.state = new_state

    def h(self, target):
        new_state = {}
        for key, value in self.state.items():
            factor = 1
            if key[target] == '1':
                factor = -1
            new_key1, new_key2 = replace_at_index(key, target)
            new_state[new_key1] = round_complex(1/sqrt(2)*value + new_state.get(new_key1, complex(0)))
            new_state[new_key2] = round_complex(factor*1/sqrt(2)*value + new_state.get(new_key2, complex(0)))
        self.state = new_state
    
    def t(self, target):
        new_state = {}
        for key, value in self.state.items():
            new_key, new_value = key, value
            if new_key[target] == '1':
                new_value = round_complex(np.exp(1j*pi/4)*new_value)
            new_state[new_key] = new_value
        self.state = new_state      
    
    def t_dag(self, target):
        new_state = {}
        for key, value in self.state.items():
            new_key, new_value = key, value
            if new_key[target] == '1':
                new_value = round_complex(np.exp(-1j*pi/4)*new_value)
            new_state[new_key] = new_value
        self.state = new_state  

    def cx(self, control, target):
        new_state = {}
        for key, value in self.state.items():
            new_key, new_value = key, value
            if new_key[control] == '1':
                new_key = flip_at_index(new_key, target)
            new_state[new_key] = new_value
        self.state = new_state
    
    def clean(self):
        new_state = {}
        for key, value in self.state.items():
            new_key, new_value = key, value
            if np.abs(value) != 0:
                new_state[new_key] = new_value
        self.state = new_state
    
    def state_out(self):
        self.clean()
        return self.state

import re

def parse_operation(s):
    """
    Parse a string operation and return a list with the operation and qubit indices.

    :param s: A string representing the operation, e.g., 'cx q[0],q[1];'
    :return: A list containing the operation name and qubit indices as integers.
    """
    # Extract operation name (assuming it's always the first word)
    operation = s.split()[0]

    # Find all occurrences of numbers in square brackets
    indices = [int(num) for num in re.findall(r'\[([0-9]+)\]', s)]

    # Combine operation with indices
    return [operation] + indices

def find_max_integer_in_string_below_n(s, n):
    """
    Find the maximum integer in a given string that is less than a specified value n.

    :param s: Input string
    :param n: The upper limit value
    :return: Maximum integer found in the string below n, or None if no such integers are present
    """
    # Extract all integers from the string
    numbers = [int(num) for num in re.findall(r'\d+', s)]

    # Filter out numbers that are greater than or equal to n
    numbers_below_n = [num for num in numbers if num < n]

    # Return the maximum number if the filtered list is not empty, otherwise return None
    return max(numbers_below_n) if numbers_below_n else None


def parser(file_str):
    operation = 0
    num_qubits = 0

    # Split the content into lines
    lines = file_str.splitlines()

    # get the number of qubits
    for line in lines:
        operation = parse_operation(line)
        if operation[0] == 'qreg':
            num_qubits = operation[1]
            break

    # create quantum circuit
    num_qubits = find_max_integer_in_string_below_n(file_str, num_qubits) + 1
    qc = QuantumSimulator(num_qubits)

    # apply operation
    for line in lines:
        operation = parse_operation(line)
        gate = operation[0]
        if gate == 'x':
            qc.x(operation[1])
        elif gate == 'cx':
            qc.cx(operation[1], operation[2])
        elif gate == 't':
            qc.t(operation[1])
        elif gate == 'tdg':
            qc.t_dag(operation[1])
        elif gate == 'h':
            qc.h(operation[1])
        
    return qc.state_out()

def simulate(file_str):
    output = parser(file_str)
    states = list(output.keys())
    state = states[0]
    decimal = int(state, 2)
    cirq_array = [0] * (2**len(state))
    cirq_array[decimal] = 1
    return cirq_array
