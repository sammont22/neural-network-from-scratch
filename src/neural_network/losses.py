import numpy as np


class MSE:

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)

        return loss

    def backward(self):
        N = self.y_pred.size

        grad_pred = (2 / N) * (self.y_pred - self.y_true)

        return grad_pred

class CategoricalCrossEntropy:

    def forward(self, y_pred, y_true):
        self.y_true = y_true

        # Avoid log(0) and division by zero
        epsilon = 1e-12
        self.y_pred = np.clip(y_pred, epsilon, 1.0)

        log_probs = np.log(self.y_pred)

        loss = -np.mean(
            np.sum(y_true * log_probs, axis=1)
        )

        return loss

    def backward(self):
        N = self.y_pred.shape[0]

        grad_pred = -self.y_true / (N * self.y_pred)

        return grad_pred