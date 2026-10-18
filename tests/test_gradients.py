import numpy as np

from neural_network.layers import Dense


def test_numerical_derivative():
    def f(x):
        return x ** 2

    x = 3.0
    epsilon = 1e-5

    # Analytical derivative
    analytical_gradient = 2 * x

    # Numerical derivative
    numerical_gradient = (
        f(x + epsilon) - f(x - epsilon)
    ) / (2 * epsilon)

    # Gradient verification
    assert np.isclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-6,
        atol=1e-7
    )


def test_dense_weight_gradient():
    X = np.array([[2.0, 3.0]])

    layer = Dense(2, 1)

    layer.weights = np.array([
        [0.5],
        [-1.0]
    ])

    layer.biases = np.array([[0.1]])

    grad_output = np.array([[3.0]])

    # 1. Analytical gradient
    layer.forward(X)
    layer.backward(grad_output)

    analytical_gradient = layer.grad_weights[0, 0]

    # 2. Numerical gradient
    epsilon = 1e-5

    original_weight = layer.weights[0, 0]

    # Positive perturbation
    layer.weights[0, 0] = original_weight + epsilon

    Z_plus = layer.forward(X)
    loss_plus = 3 * Z_plus[0, 0]

    # Negative perturbation
    layer.weights[0, 0] = original_weight - epsilon

    Z_minus = layer.forward(X)
    loss_minus = 3 * Z_minus[0, 0]

    # Restore original weight
    layer.weights[0, 0] = original_weight

    # 3. Finite differences
    numerical_gradient = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    # 4. Gradient verification
    assert np.isclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-6,
        atol=1e-7
    )