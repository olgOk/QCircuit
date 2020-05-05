# QCircuit

Small Quantum Circuit Simulator implemented on the [TensorNetwork][1]. 
--------------------------------------------------------------

## Short Overview

1. Supports JAX, TensorFlow, PyTorch, NumPy backends. By default the QCircuit uses JAX backend
to speed up calculations using GPU.
2. Supports simple visualisation of quantum circuit.
3. Implementation of the most common quantum logical gates, advanced controll gates with the ability to specify custom number of controll gates.

## Installation

``` pip install qcircuit ```

## Basic Example

Here, we build a simple two qubits quantum circuit and applying quantum gates.

```
from qcircuit import QCircuit as qc
import numpy as np

my_circuit = qc.QCircuit(2) # Create circuit on 2 qubits
my_circuit.H(0) # apply Hadamard gate on the q0
my_circuit.CX(control = [0], target = 1) # apply CX gate: q0 - controlled, q1-target

my_circuit.get_amplitude() # get amplitude measurement 
# get bitstring sampling
bitstr, max_str = my_circuit.get_bitstring()
for index in range(2 ** circuit_size):
  b = np.binary_repr(index, width=circuit_size)
  probability = bitstr[index]
  print("|" + b + "> probability " + str(probability))
  
state_vector = my_circuit.get_state_vector() # get state vector
print("state vector", state_vector)

my_circuit.visualize() # visualize the circuit
```

Please see [tutorials][2] for more examples

## Disclaimer

This library is in `alpha`. While releases will be stable enough for research, we do not recommend using this in any production environment.

[1]: https://github.com/google/TensorNetwork
[2]: https://github.com/olgOk/QCircuit/tree/master/tutorials
