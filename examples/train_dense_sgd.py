import numpy as np
import matplotlib.pyplot as plt

from neural_network.layers import Dense
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

# Model components
layer = Dense(2, 1)
loss_fn = MSE()
optimizer = SGD(learning_rate=0.01)

# Deterministic initialization
layer.weights = np.array([
    [0.1],
    [0.2]
])

layer.biases = np.array([[0.0]])

epochs = 100
loss_history = []

for epoch in range(epochs):
    # Forward propagation
    y_pred = layer.forward(X)

    # Loss calculation
    loss = loss_fn.forward(y_pred, y_true)
    loss_history.append(loss)

    # Backpropagation
    grad_output = loss_fn.backward()
    layer.backward(grad_output)

    # Update parameters
    optimizer.step(layer)

    # Training progress
    if (epoch + 1)%10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

# Evaluate after the last update
final_predictions = layer.forward(X)
final_loss = loss_fn.forward(final_predictions, y_true)

print("\nInitial loss:", loss_history[0])
print("Final loss:", final_loss)

print("\nFinal predictions:")
print(final_predictions)

# Verify improvement
assert final_loss < loss_history[0]

# Plot training loss
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Dense Layer - SGD Training")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()