# Neural Network From Scratch

A lightweight neural network library built from scratch using Python and NumPy, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The goal of this project is to understand the mathematical foundations of neural networks by implementing their core components, including forward propagation, backpropagation, activation functions, loss functions, and optimization algorithms.

## Project Objectives

This project focuses on:

- Understanding the mathematical foundations of neural networks.
- Implementing neural network layers using NumPy.
- Developing forward and backward propagation algorithms.
- Implementing activation and loss functions.
- Building gradient-based optimization algorithms.
- Training neural networks on classification problems.
- Validating mathematical correctness through automated testing.
- Comparing the implementation with PyTorch.

## Technologies

- **Python** — Core programming language.
- **NumPy** — Matrix operations and numerical computations.
- **Matplotlib** — Training visualization and analysis.
- **Pytest** — Unit testing and numerical validation.

## Project Structure

```text
neural-network-from-scratch/
│
├── src/
│   └── neural_network/
│       ├── __init__.py
│       ├── layers.py
│       ├── activations.py
│       ├── losses.py
│       ├── optimizers.py
│       └── model.py
│
├── tests/
│   └── test_layers.py
│
├── examples/
│   └── dense_example.py
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── README.md
├── .gitignore
└── pyproject.toml
```

## Implemented Features

### Dense Layer

A fully connected neural network layer supporting:

- Random weight initialization.
- Bias initialization.
- Forward propagation.
- Backward propagation.
- Gradient computation for weights, biases, and inputs.

The forward propagation is defined as:

$$
Z = XW + b
$$

The backward propagation computes:

**Weight gradients:**

$$
\frac{\partial L}{\partial W} = X^T G
$$

**Bias gradients:**

$$
\frac{\partial L}{\partial b} = \sum_{i=1}^{m} G_i
$$

**Input gradients:**

$$
\frac{\partial L}{\partial X} = GW^T
$$

where:

$$
G = \frac{\partial L}{\partial Z}
$$

These operations will form the foundation of the neural network training process.

## Installation

Clone the repository:

```bash
git clone https://github.com/sammont22/neural-network-from-scratch.git

cd neural-network-from-scratch
```

Create a virtual environment:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

## Usage

The following example demonstrates the Dense Layer implementation:

```python
import numpy as np

from neural_network.layers import Dense

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

layer = Dense(2, 3)

# Forward propagation
Z = layer.forward(X)

# Backward propagation
grad_output = np.ones_like(Z)
grad_input = layer.backward(grad_output)

print("Output shape:", Z.shape)
print("Weight gradients:", layer.grad_weights)
print("Bias gradients:", layer.grad_biases)
```

The gradient in this example is artificially initialized to ones
to demonstrate the backward propagation mechanism.

## Development Roadmap

- [x] Project initialization.
- [x] Dense Layer implementation.
- [x] Forward propagation.
- [x] Backward propagation.
- [ ] Automated unit tests.
- [ ] Gradient checking.
- [ ] Activation functions (ReLU, Sigmoid, Tanh, Softmax).
- [ ] Loss functions (MSE, Cross-Entropy).
- [ ] Gradient Descent and SGD.
- [ ] Adam optimizer.
- [ ] Neural network training pipeline.
- [ ] XOR classification.
- [ ] Handwritten digit classification.
- [ ] PyTorch comparison and benchmarking.

## Current Status

**Work in Progress — Early Development**

The project currently implements the fundamental operations
of a fully connected neural network layer.

The next development milestone focuses on automated testing
and numerical gradient verification.

## License

License to be determined.