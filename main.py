import network.neuron as neuron

# Create input layer
input_layer = neuron.Layer()
input_layer.add_neuron(neuron.Neuron(1))
input_layer.add_neuron(neuron.Neuron(2))
input_layer.add_neuron(neuron.Neuron(3))
input_layer.add_neuron(neuron.Neuron(2.5))

# Output neurons
test1 = neuron.Neuron(2)
test2 = neuron.Neuron(3)
test3 = neuron.Neuron(0.5)

# Output layer
output_layer = neuron.Layer()
output_layer.add_neuron(test1)
output_layer.add_neuron(test2)
output_layer.add_neuron(test3)

# Connect the input layer to the output layer
input_layer.connect_layer(output_layer)

# Set neuron weights
test1.set_input_weights([0.2, 0.8, -0.5, 1])
test2.set_input_weights([0.5, -0.91, 0.26, -0.5])
test3.set_input_weights([-0.26, -0.27, 0.17, 0.87])

# Grab the outputs
print(output_layer.get_outputs())