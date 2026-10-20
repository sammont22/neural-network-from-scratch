
class SGD:
    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        self.learning_rate = learning_rate

    def step(self, layer):
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.biases -= self.learning_rate * layer.grad_biases