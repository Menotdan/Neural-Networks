import network.layer as layer

input_layer = layer.Layer(4, layer.LayerEmpty)
input_layer.set_input_bias([1, 2, 3, 4])

output_layer = layer.Layer(3, layer.LayerNormal)

input_layer.connect_output_layer(output_layer)
output_layer.connect_input_layer(input_layer)

input_layer.solve()
output_layer.solve()

print(output_layer.outputs)