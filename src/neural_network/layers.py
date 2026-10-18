import numpy as np

class Dense:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        rng = np.random.default_rng(42)

        self.weights = rng.standard_normal(
            (self.input_size, self.output_size)
        )

        self.biases = np.zeros((1, self.output_size))

    def forward(self, X):
        self.X = X

        Z = X @ self.weights + self.biases

        return Z

    def backward(self, grad_output):

        self.grad_weights = self.X.T @ grad_output

        self.grad_biases = np.sum(
            grad_output,
            axis=0,
            keepdims=True
        )

        grad_input = grad_output @ self.weights.T

        return grad_input