import numpy as np
from neural_network.layers import Dense


X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

# Create a Dense Layer
layer = Dense(2, 3)

# Forward propagation
Z = layer.forward(X)

# Artificial gradient
grad_output = np.ones_like(Z)

# Backward propagation
grad_input = layer.backward(grad_output)

# Results
print("Output:")
print(Z)

print("\nGradient Weights:")
print(layer.grad_weights)

print("\nGradient Biases:")
print(layer.grad_biases)

print("\nGradient Input:")
print(grad_input)