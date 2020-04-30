#@title
import numpy as np
import re
import string
import jax
import cmath
from typing import List
import tensorflow as tf
import tensornetwork as tn
from colorama import Fore
from colorama import Style
from itertools import product


class QCircuit:
  """Implementation of a QCircuit."""

  def __init__(self, num_qubits, backend='jax'):
    self.num_qubits = num_qubits
    self.backend = backend
    # Final list that would contain all the nodes needed for network
    self.network = [self.get_initial_state()] 
    # List that would contain info about what gate is being applied for a particular qubit
    self.gate_patch = []
    # List that would contain a pair of list of control qubits and target qubit 
    self.control_gates_patch = []
    # List that contains the angle for rotation gates
    self.arguments = [None] * self.num_qubits
    self.graphics_terminal = []
    for index in range(self.num_qubits):
      self.gate_patch.append('I')
      self.graphics_terminal.append("    |  ")
      self.graphics_terminal.append("q%2s |──" % str(index))
      self.graphics_terminal.append("    |  ")
  
  # Define one qubits gate dictionary
  gates = {
    "I" : np.eye(2, dtype=np.complex128),
    
    "X" : np.array([[0.0, 1.0],
                    [1.0, 0.0]], dtype=np.complex128),
    
    "Y" : np.array([[0.0, 0.0-1.j],
                    [0.+1.j, 0.0]], dtype=np.complex128),
    
    "Z" : np.array([[1.0, 0.0],
                    [0.0, -1.0]], dtype=np.complex128),
    
    "H" : np.array([[1, 1],
                    [1, -1]], dtype=np.complex128) / np.sqrt(2),
    
    "T" : np.array([[1.0, 0.0],
                    [0.0, np.exp(1.j * np.pi / 4)]], dtype=np.complex128),
           
    "R" : np.array([[1.0, 0.0],
              [0.0, 1.0]], dtype=np.complex128),
           
    "RX" : np.array([[1.0, 1.0],
                    [1.0, 1.0]],  dtype=np.complex128),
           
    "RY" : np.array([[1.0, 1.0],
                    [1.0, 1.0]],  dtype=np.complex128),
           
    "RZ" : np.array([[1.0, 0.0],
                    [0.0, 1.0]],  dtype=np.complex128)
  }

  ########################################## GRAPHICS ############################################

  # Graphic functions to make it user friendly at the end

  colors = {
      0 : "\u001b[0m",
      1 : "\u001b[31m",
      2 : "\u001b[32m",
      3 : "\u001b[33m",
      4 : "\u001b[34m",
      5 : "\u001b[35m",
      6 : "\u001b[36m",
      7 : "\u001b[37m",
      8 : "\u001b[31m",
      9 : "\u001b[32m",
      10 : "\u001b[33m",
      11 : "\u001b[34m",
  }

  def apply_graphics_to_patch(self):
    """
      Visualize the circuit with all gates applying on it.
    """
    color_iterator = 1
    for control_gate in self.control_gates_patch:
      full_list = control_gate[0] + [control_gate[1]]
      for qubit in full_list:
        if any(qubit - it_qubit == 1 for it_qubit in full_list):
          self.graphics_terminal[qubit * 3] += "%s╔╩╗%s   " % (self.colors[color_iterator], self.colors[0])
        else:
          self.graphics_terminal[qubit * 3] += "%s╔═╗%s   " % (self.colors[color_iterator], self.colors[0])
        if (qubit == control_gate[1]):
          self.graphics_terminal[qubit * 3 + 1] += "%s║%s║%s───" % (self.colors[color_iterator], self.gate_patch[qubit][1], self.colors[0])
        else:
          self.graphics_terminal[qubit * 3 + 1] += "%s║%s║%s───" % (self.colors[color_iterator], self.gate_patch[qubit][1].lower(), self.colors[0])
        if any(qubit - it_qubit == -1 for it_qubit in full_list):
          self.graphics_terminal[qubit * 3 + 2] += "%s╚╦╝%s   " % (self.colors[color_iterator], self.colors[0])
        else:
          self.graphics_terminal[qubit * 3 + 2] += "%s╚═╝%s   " % (self.colors[color_iterator], self.colors[0])
      color_iterator = color_iterator + 1

    for qubit in range(self.num_qubits):
      if (self.gate_patch[qubit] == 'I'):
        self.graphics_terminal[qubit * 3] +=     "      "
        self.graphics_terminal[qubit * 3 + 1] += "──────"
        self.graphics_terminal[qubit * 3 + 2] += "      "
      elif ("Target" not in self.gate_patch[qubit] and "Control" not in self.gate_patch[qubit]):
        self.graphics_terminal[qubit * 3] +=     "╔═╗   "
        self.graphics_terminal[qubit * 3 + 1] += "║%s║───" % self.gate_patch[qubit]
        self.graphics_terminal[qubit * 3 + 2] += "╚═╝   "


  ########################################## GRAPHICS ############################################


  # Methods of the QCircuit class

  def get_initial_state(self):
    """"
      Generate and returns the node of the initial state of the quantum circuit.
    """

    if self.num_qubits <= 0 or not isinstance(self.num_qubits, int):
      raise ValueError("Amount of qubits should be not-negative integer.")
    # create initial state vector 
    initial_state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
    initial_state[0] = 1.0 + 0.j
    initial_state = np.transpose(initial_state)
    # wrap the tensor of the initial state in to a node
    initial_state_node = tn.Node(initial_state, backend=self.backend)
    return initial_state_node


  # Generate two quibits gates
  def generate_control_gate(self, control, target:List, gate:str):
    """
      Generate and return the tensor of the control gate for any system
      of different number of qubits with the consideration of the given
      control and target qubits.
      Args:
        control: The index of the control qubit
        target: The index of the target qubit
        gate: a type of gate to be generated (X, Y, etc.)
      Returns:
        A tensor of the contol gate
    """
    control_gate = np.eye(2**self.num_qubits, dtype=np.complex128)
    tuples = []
    # Searches for all the numbers up to 2**self.num_qubits such that in 
    # binary representation they have '1' in control-place and '0' in 
    # target place. 
    for i in range(2**self.num_qubits):
      if not (i & (1 << target)) and all(i & (1 << control_qubit) for control_qubit in control):
        swap = i + 2**target
        # Embeds the transformation into the matrix.
        control_gate[i][i] = self.gates[gate][0][0]
        control_gate[i][swap] = self.gates[gate][0][1]
        control_gate[swap][i] = self.gates[gate][1][0]
        control_gate[swap][swap] = self.gates[gate][1][1]
    # If control gate applies Hadamard gate, puts the whole system into
    # superposition.
    if gate == 'H':
      control_gate = control_gate * (1. + 1.j) / np.sqrt(2)
    return control_gate

  def apply_arguments(self, gate):
    """
        Applies R, RX, RY, RZ gates on quantum state.
        Args:
          gate: number of a qubit to apply gate on
        Returns:
          None. 
    """
    if self.gate_patch[gate] == 'R':
        self.gates['R'][1][1] = self.arguments[gate]
    if self.gate_patch[gate] == 'RX':
      self.gates['RX'][0][0] = np.cos(self.arguments[gate])
      self.gates['RX'][1][1] = np.cos(self.arguments[gate])
      self.gates['RX'][1][0] = -1.j*np.sin(self.arguments[gate])
      self.gates['RX'][0][1] = -1.j*np.sin(self.arguments[gate])
    if self.gate_patch[gate] == 'RY':
      self.gates['RY'][0][0] = np.cos(self.arguments[gate])
      self.gates['RY'][1][1] = np.cos(self.arguments[gate])
      self.gates['RY'][1][0] = np.sin(self.arguments[gate])
      self.gates['RY'][0][1] = -np.sin(self.arguments[gate])
    if self.gate_patch[gate] == 'RZ':
      self.gates['RZ'][0][0] = np.exp(-1.j*self.arguments[gate])
      self.gates['RZ'][1][1] = np.exp(1.j*self.arguments[gate])


  def evaluate_patch(self):
    """
      Evaluate the gates applying on the curcuit at the given moment of time.
      The tensor of the correcponding gates stored in a tensornetwork node. 
    """
    if all(self.gate_patch[i] == 'I' for i in range(self.num_qubits)):
      return

    # Call graphic function
    self.apply_graphics_to_patch()
  
    # Create matrix for all control gates in the current patch
    for control_gate_info in self.control_gates_patch:
      target_qubit = control_gate_info[1]
      if self.gate_patch[target_qubit][1] == 'R':
        self.gates['R'][1][1] = self.arguments[target_qubit]
      control_gate = self.generate_control_gate(control_gate_info[0], target_qubit, self.gate_patch[target_qubit][1])
      control_gate = control_gate.transpose()
      self.network.append(tn.Node(control_gate, backend=self.backend))
      self.gate_patch[target_qubit] = 'I'
      for qubit in control_gate_info[0]:
        self.gate_patch[qubit] = 'I'
    self.control_gates_patch = []

    self.apply_arguments(self.num_qubits - 1)
    result_matrix = self.gates[self.gate_patch[self.num_qubits - 1]]

    # expand space using tensor product
    shape = 4
    for gate in reversed(range(self.num_qubits - 1)):
      self.apply_arguments(gate)
      result_matrix = np.tensordot(result_matrix, self.gates[self.gate_patch[gate]], axes=0)
      result_matrix = result_matrix.transpose((0, 2, 1, 3)).reshape((shape, shape))
      shape = len(result_matrix) * 2
    result_matrix = result_matrix.transpose()
    # store the moment in the node and append to the curcuit
    self.network.append(tn.Node(result_matrix, backend=self.backend))
    for index in range(self.num_qubits):
      self.gate_patch[index] = 'I'
      self.arguments[index] = None

  def get_state_vector(self):
    """
      Returns resulting state vector as a tensor of rank 1.
      Round values to 3 decimal points.
    """
    # connect all nodes and evaluate all tensors stored in it
    self.evaluate_patch()

    if len(self.network) > 1:
      for index in reversed(range(1, len(self.network) - 1)):
        self.network[index + 1][0] ^ self.network[index][1]
      self.network[1][0] ^ self.network[0][0]
    nodes = tn.reachable(self.network[1])
    result = tn.contractors.greedy(nodes, ignore_edge_order=True)
    # round the result to three decimals
    state_vecor = np.round(result.tensor, 3)
    return state_vecor

  # Get amplitude
  def get_amplitude(self):
    """
      Print amplitudes of the final state vector of the circuit.
      Amplitudes defined as the length of the state vector on Bloch sphere.
      Round values to 3 decimal points.
    """

    state_vector = self.get_state_vector()
    # amplitude = sqrt( (real_part)^2 + (complex_part)^2)
    for index in range(2 ** self.num_qubits):
      amplitude = np.absolute(state_vector[index])
      # decimal to binary
      b = np.binary_repr(index, width=self.num_qubits)
      print("|" + b + "> amplitude " + str(amplitude))
      

  # Get bitstring
  def get_bitstring(self):
    """
      Print bitstring for the final state vector of the circuit.
      Probability calculated as a value times value_conjugate.
      Returns:
        Probability of each bit.
        Binary reprsentation of the most probabal bitstring.
    """

    state_vector = self.get_state_vector()
    sample = {}
    # probability = complex_magnitude * complex_magnitude_conjugate
    for index in range(2 ** self.num_qubits):
      probability = state_vector[index] * np.conjugate(state_vector[index])
      probability = np.round(np.real(probability), 3)
      b = np.binary_repr(index, width=self.num_qubits)
      sample[index] = probability
      # print("|" + b + "> probability " + str(probability))
    return sample, np.binary_repr(max(sample, key=sample.get), width=self.num_qubits)

  # Get visualization
  def visualize(self):
    """
      Visualize the quantum circuit.
    """
    self.evaluate_patch()
    for string in self.graphics_terminal:
      print (string)

  # checks for the correct input
  def check_input_one_gate(self, target:int):
    """"
      Check for the basics inputs of the one-qubit gates.
      Args:
        target: a target qubit.
      Return: None.
      Raise:
        Value Errors.
    """
    if target > self.num_qubits - 1:
      raise ValueError("Qubit's index exceed the specified size of the cirquit.")
    if target < 0 or not isinstance(target, int):
      raise ValueError("Target gate should be not-negative integer.")

  
  # Add one qubit gates
  def X(self, target:int):
    """Add X gate (logical NOT) to the stack of current moment.
      Args:
        target: An index of a qubit node on which X gate acts.
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    # if gates applied on all quibits, evaluate current moment and start
    # to fill out next moment
    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    self.gate_patch[target] = 'X'
    

  def Y(self, target:int):
    """Add Y gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which Y gate acts
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    self.gate_patch[target] = 'Y'


  def Z(self, target:int):
    """Add Z gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which Z gate acts
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    self.gate_patch[target] = 'Z'


  def H(self, target:int):
    """Add H gate (Hadamard Gate) to the stack of current moment.
      Hadamara Gate brings the initial state vector to its superposition state.
      Args:
        target: An index of a qubit node on which H gate acts
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    self.gate_patch[target] = 'H'


  def T(self, target:int):
    """Add T gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which T gate acts
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    self.gate_patch[target] = 'T'

  def R(self, phi:float, target:int):
    """Add R gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which R gate acts.
        phi: an angle in radians which corresponds to the rotation of
        the qubit state around the z axis by the given value of phi.
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    # store the value(s) of angle passed
    self.arguments[target] = np.exp(1.j * phi)
    self.gate_patch[target] = 'R'

# = = = = = = = = = == = = = = = = = = = == = = = = = = = = = = = = = = = = = = == = = 

  def RX(self, phi:float, target:int):
    """Add RX gate to the stack of current moment.
       Args:
          target: An index of a qubit node on which R gate acts.
          phi: an angle in radians which corresponds to the rotation of
               the qubit state around the X axis by the given value of phi.
       Returns: None.
       Raise: ValueError if an index of the target quibit out of circuit size.
              ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
     # store the value(s) of angle passed
    self.arguments[target] = phi / 2
    self.gate_patch[target] = 'RX'

  def RY(self, phi:float, target:int):
    """Add RY gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which R gate acts.
        phi: an angle in radians which corresponds to the rotation of
        the qubit state around the Y axis by the given value of phi.
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    # store the value(s) of angle passed
    self.arguments[target] = phi / 2
    self.gate_patch[target] = 'RY'

  def RZ(self, phi:float, target:int):
    """Add RZ gate to the stack of current moment.
      Args:
        target: An index of a qubit node on which R gate acts.
        phi: an angle in radians which corresponds to the rotation of
        the qubit state around the Z axis by the given value of phi.
      Returns: None.
      Raise: ValueError if an index of the target quibit out of circuit size.
             ValueError if target quibit is a float or negative number.
    """

    self.check_input_one_gate(target)

    if (self.gate_patch[target] != 'I'):
      self.evaluate_patch()
    # store the value(s) of angle passed
    self.arguments[target] = phi / 2
    self.gate_patch[target] = 'RZ'

# = = = = = = = = = == = = = = = = = = = == = = = = = = = = = = = = = = = = = = == = = 

  # checks for incorrect arguments for many qubits gate
  def check_input_control_gate(self, control:List, target:int):
    """"
      Check for the basics inputs of the controll gates.
      Args:
        control: a list of the controlled qubits.
        target: a target qubit.
      Return: None.
      Raise:
        Value Errors.
    """
    if not isinstance(control, list):
      raise ValueError("Control must be a list.")
    if not len(control):
      raise ValueError("No control qubits has been provided.")
    if target > self.num_qubits - 1:
      raise ValueError("Qubit's index exceed the specidied size of the cirquit.")
    if target < 0 or not isinstance(target, int):
      raise ValueError("Target gate should be not-negative integer.")
    for control_qubit in control:
      if control_qubit > self.num_qubits - 1:
        raise ValueError("Qubit's index exceed the specidied size of the cirquit.")
      if control_qubit < 0 or not isinstance(control_qubit, int):
        raise ValueError("Control gate should be not-negative integer.")
    if target in control:
      raise ValueError("Target qubit was sent as a control.")
    if (not len(set(control)) == len(control)):
      raise ValueError("Control list contains repeating elements.")

  # Add two qubits gates

  def CX(self, control:List, target:int):
    """Add CX (CNOT) gate to the stack of current moment.
      Args:
        control: An indices of qubits that serve as a control elements
        target: An index of a qubit node on which CX gate acts
      Returns: None.
      Raise: ValueError if indices of quibit out of circuit size.
             ValueError if target and control indeces are equal.
             ValueError if target or control quibit is a float or negative number.
    """

    self.check_input_control_gate(control, target)
    # if gates applied on all quibits, evaluate current moment and start
    # to fill out next moment
    if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
      self.evaluate_patch()
    self.gate_patch[target] = 'CX_Target_' + str(target)
    for control_qubit in control:
      self.gate_patch[control_qubit] = 'CX_Control_' + str(target)
    self.check_input_control_gate(control, target)
    self.control_gates_patch.append((control, target))

  def CZ(self, control:List, target:int):
    """Add CZ gate to the stack of current moment.
      Args:
        control: An indices of qubits that serve as a control elements
        target: An index of a qubit node on which CZ gate acts
      Returns: None.
      Raise: ValueError if indices of quibit out of circuit size.
             ValueError if target and control indeces are equal.
             ValueError if target or control quibit is a float or negative number.
    """

    self.check_input_control_gate(control, target)
    if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
      self.evaluate_patch()
    self.gate_patch[target] = 'CZ_Target_' + str(target) 
    for control_qubit in control:
      self.gate_patch[control_qubit] = 'CZ_Control_' + str(target)
    self.control_gates_patch.append((control, target))


  def CY(self, control:List, target:int):
    """Add CY gate to the stack of current moment.
      Args:
        control: An indices of qubits that serve as a control elements
        target: An index of a qubit node on which CY gate acts
      Returns: None.
      Raise: ValueError if indices of quibit out of circuit size.
             ValueError if target and control indeces are equal.
             ValueError if target or control quibit is a float or negative number.
    """

    self.check_input_control_gate(control, target)
    if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
      self.evaluate_patch()
    self.gate_patch[target] = 'CY_Target_' + str(target)
    for control_qubit in control:
      self.gate_patch[control_qubit] = 'CY_Control_' + str(target)
    self.control_gates_patch.append((control, target))

  def CH(self, control:List, target:int):
    """Add CH gate to the stack of current moment.
      Args:
        control: An indices of qubits that serve as a control elements
        target: An index of a qubit node on which CH gate acts
      Returns: None.
      Raise: ValueError if indices of quibit out of circuit size.
             ValueError if target and control indeces are equal.
             ValueError if target or control quibit is a float or negative number.
    """

    self.check_input_control_gate(control, target)
    if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
      self.evaluate_patch()

    self.gate_patch[target] = 'CH_Target_' + str(target)
    for control_qubit in control:
      self.gate_patch[control_qubit] = 'CH_Control_' + str(target)
    self.control_gates_patch.append((control, target))


  def CR(self, phi:float, control:List, target:int):
    """Add CR gate to the stack of current moment.
      Args:
        phi: an angle in radians which corresponds to the rotation of
        the qubit state around the z axis by the given value of phi.
        control: An indices of qubits that serve as a control elements
        target: An index of a qubit node on which CR gate acts
      Returns: None.
      Raise: ValueError if indices of quibit out of circuit size.
             ValueError if target and control indeces are equal.
             ValueError if target or control quibit is a float or negative number.
    """

    self.check_input_control_gate(control, target)

    if (self.gate_patch[target] != 'I' or self.gate_patch[control] != 'I'):
      self.evaluate_patch()

    self.arguments[target] = np.exp(1.j * phi, dtype=np.complex128)
    self.gate_patch[target] = 'CR_Target_' + str(target)
    for control_qubit in control:
      self.gate_patch[control_qubit] = 'CR_Control_' + str(target)
    self.control_gates_patch.append((control,target))



  # Create quantum Oracle. Needed for Deutch Algorithm
  def Uf(self, func:callable):
    """
      Create a unitary matrix which is equvalent to the function passed
      as an argument.
      Args:
        func: a binary function which has to be converted to unitary matrix.
      Raise: ValueError if argument is not callable. 
    """
    if not callable(func):
      raise ValueError("Argument must be a function.")
  
    self.evaluate_patch()

    size = 2 ** self.num_qubits
    U = np.zeros((size,size), dtype=np.complex128)

    # convert binary state-vector to integer
    def bin2int(bits):
      integer = 0
      for shift, j in enumerate(bits[::-1]):
          if j:
              integer += 1 << shift
      return integer

    # iterate through each state and build the unitary matrix
    for state in product({0,1}, repeat=self.num_qubits):
      x = state[:~0]
      y = state[~0]
      unitary_value = y ^ func(*x) #bitwise logical XOR
      i = bin2int(state)
      j = bin2int(list(x) + [unitary_value])
      U[i, j] = 1.0 + 0.j
    
    # add to the network 
    self.network.append(tn.Node(U, backend=self.backend))
