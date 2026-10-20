import numpy as np

class ReLU:

    def forward(self, X):
        self.X = X

        return np.maximum(0, X)

    def backward(self, grad_output):
        grad_input = grad_output * (self.X > 0)

        return grad_input

class Sigmoid:

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, grad_output):
        sigmoid_derivative = self.output * (1 - self.output)

        grad_input = grad_output * sigmoid_derivative

        return grad_input

class Tanh:

    def forward(self, X):
        self.output = np.tanh(X)

        return self.output

    def backward(self, grad_output):
        tanh_derivative = 1 - self.output ** 2

        grad_input = grad_output * tanh_derivative

        return grad_input

class Softmax:

    def forward(self, X):
        shifted_X = X - np.max(X, axis=1, keepdims=True)

        exp_X = np.exp(shifted_X)

        self.output = exp_X / np.sum(
            exp_X,
            axis=1,
            keepdims=True
        )

        return self.output

    def backward(self, grad_output):
        weighted_gradient = np.sum(
            grad_output * self.output,
            axis=1,
            keepdims=True
        )

        grad_input = self.output * (
            grad_output - weighted_gradient
        )

        return grad_input