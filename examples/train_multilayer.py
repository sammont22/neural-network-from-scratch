import numpy as np

from neural_network.layers import Dense
from neural_network.activations import ReLU
from neural_network.losses import MSE
from neural_network.optimizers import SGD


# Dataset
X = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

y_true = np.array([
    [1.0],
    [2.0]
])

# Network
dense1 = Dense(2, 4)
relu = ReLU()
dense2 = Dense(4, 1)

# Deterministic initialization for this experiment
dense1.weights = np.array([
    [0.1, 0.2, 0.1, 0.2],
    [0.2, 0.1, 0.2, 0.1]
])
dense1.biases = np.zeros((1, 4))

dense2.weights = np.array([
    [0.1],
    [0.2],
    [0.1],
    [0.2]
])
dense2.biases = np.zeros((1, 1))

loss_fn = MSE()
optimizer = SGD(learning_rate=0.01)

epochs = 100
loss_history = []

for epoch in range(epochs):

    # Forward propagation
    Z1 = dense1.forward(X)
    A1 = relu.forward(Z1)
    y_pred = dense2.forward(A1)

    # Loss
    loss = loss_fn.forward(y_pred, y_true)
    loss_history.append(float(loss))

    # Backpropagation (reverse order)
    grad = loss_fn.backward()
    grad = dense2.backward(grad)
    grad = relu.backward(grad)
    grad = dense1.backward(grad)

    # Update both Dense layers
    optimizer.step(dense1)
    optimizer.step(dense2)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

# Final evaluation
Z1 = dense1.forward(X)
A1 = relu.forward(Z1)
final_predictions = dense2.forward(A1)

final_loss = loss_fn.forward(final_predictions, y_true)

print("\nInitial loss:", loss_history[0])
print("Final loss:", final_loss)
print("\nFinal predictions:")
print(final_predictions)

assert final_loss < loss_history[0]