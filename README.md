# Neural Network From Scratch

A compact neural network library implemented with Python and NumPy, without relying on automatic differentiation or deep learning frameworks.

This project builds the core training pipeline explicitly—from tensor operations and analytical gradients to parameter updates—so that every stage of forward propagation and backpropagation remains inspectable. It is designed as an AI/ML engineering portfolio project focused on mathematical correctness, clean component boundaries, and reproducible experiments.

## Highlights

- Fully connected (`Dense`) layers with analytical forward and backward passes
- ReLU, Sigmoid, Tanh, and numerically stable Softmax activations
- Mean Squared Error and Categorical Cross-Entropy losses
- Stochastic Gradient Descent parameter updates
- Sequential `NeuralNetwork` API for inference and full-batch training
- Finite-difference gradient checks for layers, activations, and losses
- End-to-end XOR experiment with a plotted decision boundary
- 32 automated tests covering numerical behavior and training flow

## Architecture

### Dense layer

For an input batch \(X\), weights \(W\), and biases \(b\), the forward pass computes:

$$
Z = XW + b
$$

Given an upstream gradient \(G = \partial L / \partial Z\), the backward pass computes:

$$
\frac{\partial L}{\partial W} = X^T G, \qquad
\frac{\partial L}{\partial b} = \sum_i G_i, \qquad
\frac{\partial L}{\partial X} = GW^T
$$

`Dense.backward()` stores the weight and bias gradients on the layer and returns the gradient with respect to its input.

### Activation functions

Each activation exposes `forward()` and `backward()` methods:

- **ReLU** applies \(\max(0, x)\) and masks gradients for non-positive inputs.
- **Sigmoid** maps values to \((0, 1)\) and uses \(\sigma(x)(1-\sigma(x))\) in its backward pass.
- **Tanh** maps values to \((-1, 1)\) and uses \(1-\tanh^2(x)\) for its derivative.
- **Softmax** converts each row of logits into a probability distribution. The implementation subtracts the row maximum before exponentiation for numerical stability and computes the vector-Jacobian product during backpropagation.

### Loss functions

- **MSE** returns the mean squared error over all prediction elements and its analytical gradient.
- **CategoricalCrossEntropy** supports one-hot encoded multiclass targets. Predicted probabilities are clipped before the logarithm and division operations to avoid undefined numerical values.

### Optimization

`SGD` applies the update rule:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

where \(\eta\) is a positive learning rate. The optimizer updates the weights and biases of each trainable `Dense` layer.

### Sequential model

`NeuralNetwork` composes layers in order and provides:

- `forward(X)` — passes a batch through every layer.
- `backward(grad_output)` — propagates gradients through the layers in reverse order.
- `step(optimizer)` — updates every `Dense` layer through the supplied optimizer.
- `fit(X, y_true, loss_fn, optimizer, epochs)` — runs full-batch forward propagation, loss evaluation, backpropagation, and parameter updates while returning the loss history.
- `predict(X)` — returns the raw output of a forward pass.

## XOR Example

The end-to-end example trains the following network on the four XOR samples:

```text
2 inputs -> Dense(2, 4) -> Tanh -> Dense(4, 1) -> Sigmoid
```

It uses MSE, SGD with a learning rate of `0.5`, and `6000` full-batch epochs. With the deterministic layer initialization currently used by the project, the run produces:

```text
Initial loss: 0.28887210054290374
Final loss:   0.00023680718339083806

Predictions:
[[0.00717173]
 [0.98233585]
 [0.98487555]
 [0.01884208]]

Predicted classes:
[[0]
 [1]
 [1]
 [0]]

Training accuracy: 1.0
```

The script also evaluates the trained model on a two-dimensional grid and displays the learned decision regions and the `0.5` decision boundary with Matplotlib.

> These metrics are training results on the complete four-sample XOR dataset, not held-out evaluation results.

## Gradient Checking

Analytical gradients are validated against centered finite-difference approximations:

$$
\frac{\partial f}{\partial x} \approx
\frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}
$$

The test suite checks gradients for Dense weights, biases, and inputs; ReLU, Sigmoid, Tanh, and Softmax; MSE; and Categorical Cross-Entropy. These checks complement direct unit tests for expected values, shapes, optimizer steps, and model training behavior.

## Installation

Requirements:

- Python 3.10 or newer
- NumPy 1.26 or newer
- Pytest for the test suite
- Matplotlib for the training and XOR visualizations

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/sammont22/neural-network-from-scratch.git
cd neural-network-from-scratch

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest matplotlib
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Running the Project

Run all automated tests:

```bash
python -m pytest -q
```

Run the XOR experiment and open its decision-boundary visualization:

```bash
python examples/xor.py
```

Additional examples:

```bash
python examples/dense_example.py
python examples/train_dense_sgd.py
python examples/train_multilayer.py
```

## Project Structure

```text
neural-network-from-scratch/
├── examples/
│   ├── dense_example.py       # Dense forward/backward demonstration
│   ├── train_dense_sgd.py     # Single-layer regression training loop
│   ├── train_multilayer.py    # Manual multilayer training loop
│   └── xor.py                 # End-to-end XOR training and decision boundary
├── src/neural_network/
│   ├── activations.py         # ReLU, Sigmoid, Tanh, and Softmax
│   ├── layers.py              # Dense layer
│   ├── losses.py              # MSE and CategoricalCrossEntropy
│   ├── model.py               # Sequential NeuralNetwork pipeline
│   └── optimizers.py          # SGD optimizer
├── tests/
│   ├── test_activations.py
│   ├── test_gradients.py
│   ├── test_layers.py
│   ├── test_losses.py
│   ├── test_model.py
│   └── test_optimizers.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Current Scope and Limitations

The current version intentionally keeps the training system small and transparent:

- Training in `NeuralNetwork.fit()` is full-batch; there is no data loader, batching, or shuffling.
- SGD is the only optimizer and does not yet include momentum, weight decay, or learning-rate scheduling.
- The model does not provide serialization, callbacks, evaluation metrics, or train/validation splitting.
- Categorical Cross-Entropy expects one-hot encoded targets and probabilities supplied by a separate Softmax layer.
- The XOR result demonstrates optimization on a tiny training set and is not a generalization benchmark.
- The project is educational and is not intended as a production replacement for PyTorch or TensorFlow.

## Roadmap

The following items are planned rather than currently implemented:

- SGD with Momentum and the Adam optimizer
- Mini-batch training, shuffling, and validation support
- A handwritten-digit classification experiment
- Equivalent PyTorch models for correctness and performance benchmarking
- Model persistence and richer training diagnostics

## License

No license has been selected yet.
