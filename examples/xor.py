import numpy as np
import matplotlib.pyplot as plt

from neural_network.layers import Dense
from neural_network.activations import Tanh, Sigmoid
from neural_network.losses import MSE
from neural_network.optimizers import SGD
from neural_network.model import NeuralNetwork


# XOR dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y_true = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# Build neural network
model = NeuralNetwork([
    Dense(2, 4),
    Tanh(),
    Dense(4, 1),
    Sigmoid()
])

# Training configuration
loss_fn = MSE()
optimizer = SGD(learning_rate=0.5)

# Train model
history = model.fit(
    X,
    y_true,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=6000
)

# Predictions
predictions = model.predict(X)

# Convert probabilities into binary classes
predicted_classes = (predictions >= 0.5).astype(int)

# Final evaluation
final_loss = loss_fn.forward(predictions, y_true)
accuracy = np.mean(predicted_classes == y_true)

print("Initial loss:", history[0])
print("Final loss:", final_loss)

print("\nPredictions:")
print(predictions)

print("\nPredicted classes:")
print(predicted_classes)

print("\nAccuracy:", accuracy)

assert np.array_equal(predicted_classes, y_true.astype(int))

# Create a grid of points
x1 = np.linspace(-0.25, 1.25, 200)
x2 = np.linspace(-0.25, 1.25, 200)

xx, yy = np.meshgrid(x1, x2)

grid_points = np.column_stack([
    xx.ravel(),
    yy.ravel()
])

# Predict probabilities on the grid
grid_predictions = model.predict(grid_points)
Z = grid_predictions.reshape(xx.shape)

# Plot decision regions
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    levels=np.linspace(0, 1, 21),
    alpha=0.7
)

# Decision boundary at probability 0.5
plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    linewidths=2
)

# Original XOR samples
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y_true.ravel(),
    s=150,
    edgecolors="black"
)

plt.colorbar(label="Predicted probability")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("XOR - Neural Network Decision Boundary")

plt.tight_layout()
plt.show()