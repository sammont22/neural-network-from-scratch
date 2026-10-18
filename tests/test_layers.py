import numpy as np

from neural_network.layers import Dense


def test_dense_forward_shape():

    X = np.random.default_rng(42).standard_normal((32, 5))

    layer = Dense(5, 10)

    Z = layer.forward(X)

    assert Z.shape == (32, 10)

def test_dense_forward_values():

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    layer = Dense(2, 2)

    layer.weights = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    layer.biases = np.array([
        [1.0, 1.0]
    ])

    Z = layer.forward(X)

    expected_Z = np.array([
        [8.0, 11.0],
        [16.0, 23.0]
    ])

    assert np.allclose(Z, expected_Z)

def test_dense_backward_values():

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    layer = Dense(2, 2)

    layer.weights = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    layer.biases = np.array([
        [1.0, 1.0],
    ])

    Z = layer.forward(X)

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    grad_input = layer.backward(grad_output)

    # Expected gradients
    expected_grad_weights = np.array([
        [10.0, 14.0],
        [14.0, 20.0]
    ])

    expected_grad_biases = np.array([
        [4.0, 6.0],
    ])

    expected_grad_input = np.array([
        [5.0, 11.0],
        [11.0, 25.0]
    ])

    # Numerical verification
    assert np.allclose(layer.grad_weights, expected_grad_weights)
    assert np.allclose(layer.grad_biases, expected_grad_biases)
    assert np.allclose(grad_input, expected_grad_input)