import numpy as np
import random

LayerEmpty = 0
LayerNormal = 1

class Layer:
    def __init__(self, neurons, l_type):
        self.neuron_count = neurons
        self.weights = []
        self.inputs = []
        self.biases = []
        self.previous_layer = None
        self.next_layer = None
        self.outputs = []
        self.l_type = l_type

        for i in range(self.neuron_count):
            self.biases.append(1) # All biases are 1 by default

    # This sets the output for a layer by changing the biases on an empty layer
    def set_input_bias(self, input_data):
        for i in range(len(input_data)):
            self.biases[i] = input_data[i]

    # Connect an input
    def connect_input_layer(self, new_input):
        self.previous_layer = new_input
        
        # Create weights for each input
        for i in range(self.neuron_count):
            new_weights = []
            if self.l_type == 1: # Normal layer
                for x in range(new_input.neuron_count):
                    new_weights.append(random.randint(-10, 10)) # Random weights
            else: # Empty layer
                for x in range(new_input.neuron_count):
                    new_weights.append(0) # 1 weight means no change
            self.weights.append(new_weights)
    
    # Connect an output
    def connect_output_layer(self, new_output):
        self.next_layer = new_output

    # Solve the layer
    def solve(self):
        if self.previous_layer != None:
            for i in range(self.previous_layer.neuron_count):
                self.inputs.append(self.previous_layer.outputs[i])
            self.outputs = np.dot(self.weights, self.inputs) + self.biases
        else:
            self.outputs = np.array(self.biases)

        print(self.outputs)
        self.inputs = []