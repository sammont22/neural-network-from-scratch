import numpy as np
import pytest

from neural_network.layers import Dense
from neural_network.optimizers import SGD


def test_sgd_step():
    layer = Dense(2, 1)

    layer.weights = np.array([
        [0.5],
        [-1.0]
    ])

    layer.biases = np.array([[0.1]])

    X = np.array([[2.0, 3.0]])
    grad_output = np.array([[3.0]])

    # Calculate gradients
    layer.forward(X)
    layer.backward(grad_output)

    # Update parameters
    optimizer = SGD(learning_rate=0.01)
    optimizer.step(layer)

    expected_weights = np.array([
        [0.44],
        [-1.09]
    ])

    expected_biases = np.array([[0.07]])

    np.testing.assert_allclose(
        layer.weights,
        expected_weights
    )

    np.testing.assert_allclose(
        layer.biases,
        expected_biases
    )

@pytest.mark.parametrize("learning_rate", [0, -0.01])
def test_sgd_invalid_learning_rate(learning_rate):

    with pytest.raises(ValueError):
        SGD(learning_rate=learning_rate)

def test_sgd_multiple_steps():

    layer = Dense(2, 1)

    layer.weights = np.array([
        [0.5],
        [-1.0]
    ])

    layer.biases = np.array([[0.1]])

    layer.grad_weights = np.array([
        [6.0],
        [9.0]
    ])

    layer.grad_biases = np.array([[3.0]])

    optimizer = SGD(learning_rate=0.01)

    optimizer.step(layer)
    optimizer.step(layer)

    expected_weights = np.array([
        [0.38],
        [-1.18]
    ])

    expected_biases = np.array([[0.04]])

    np.testing.assert_allclose(
        layer.weights,
        expected_weights
    )

    np.testing.assert_allclose(
        layer.biases,
        expected_biases
    )