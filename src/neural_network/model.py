from .layers import Dense


class NeuralNetwork:

    def __init__(self, layers):
        if not layers:
            raise ValueError(
                "A neural network must contain at least one layer"
            )

        self.layers = list(layers)

    def forward(self, X):
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_output):
        grad = grad_output

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def step(self, optimizer):
        for layer in self.layers:
            if isinstance(layer, Dense):
                optimizer.step(layer)

    def fit(self, X, y_true, loss_fn, optimizer, epochs=100):
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer")

        loss_history = []

        for epoch in range(epochs):

            # Forward propagation
            y_pred = self.forward(X)

            # Loss calculation
            loss = loss_fn.forward(y_pred, y_true)
            loss_history.append(float(loss))

            # Backpropagation
            grad_output = loss_fn.backward()
            self.backward(grad_output)

            # Parameter update
            self.step(optimizer)

        return loss_history

    def predict(self, X):
        return self.forward(X)