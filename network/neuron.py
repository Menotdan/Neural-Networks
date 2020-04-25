import numpy as np

class Neuron:
    def __init__(self, bias):
        self.inputs = []
        self.bias = bias
    def get_output(self):
        ret = 0
        local_inputs = []
        local_weights = []
        for i in range(len(self.inputs)):
            local_weights.append(self.inputs[i][1])
            local_inputs.append(self.inputs[i][0].get_output())

        ret = np.dot(local_inputs, local_weights) + self.bias

        return ret
    def add_input(self, input_neuron, weight):
        self.inputs.append((input_neuron, weight))
    def set_input_weights(self, weights):
        if len(weights) > len(self.inputs):
            return
        for i in range(len(weights)):
            self.inputs[i] = (self.inputs[i][0], weights[i])

class Layer:
    def __init__(self):
        self.neurons = []
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
    def connect_neuron(self, neuron):
        for n in self.neurons:
            neuron.add_input(n, 1)
    def connect_layer(self, layer):
        for ln in layer.neurons:
            self.connect_neuron(ln)
    def get_outputs(self):
        ret = []
        for n in self.neurons:
            ret.append(n.get_output())
        return ret