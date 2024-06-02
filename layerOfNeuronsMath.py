import numpy as np

# Each neuron in a layer takes exactly the same input, but contains its own set of weights and its own bias, 
# producing its own unique output. Here, we initialize the parameters randomly
# Bias: offset the output positively or negatively
# Weights: trainable factor of how much of this input to use 

# A single neuron
inputsSingle = [1.0, 2.0, 3.0]
weightsSingle= [0.2, 0.4, 1.2]  
biasSingle = 2.0 

outputsSingle = np.dot(weightsSingle, inputsSingle) + biasSingle
print(outputsSingle)

# Multiple neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
 # Zeroed output of given neuron
    neuron_output = 0
 # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
 # Multiply this input by associated weight
 # and add to the neuron's output variable
        neuron_output += n_input*weight
 # Add bias
    neuron_output += neuron_bias
 # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)
print(layer_outputs)