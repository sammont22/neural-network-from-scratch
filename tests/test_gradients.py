import numpy as np

from neural_network.layers import Dense
from neural_network.activations import ReLU, Sigmoid, Tanh, Softmax
from neural_network.losses import MSE, CategoricalCrossEntropy


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

def finite_difference_gradient(array, loss_fn, epsilon=1e-5):

    numerical_gradient = np.zeros_like(array, dtype=float)

    for index in np.ndindex(array.shape):

        original_value = array[index]

        # Positive perturbation
        array[index] = original_value + epsilon
        loss_plus = loss_fn()

        # Negative perturbation
        array[index] = original_value - epsilon
        loss_minus = loss_fn()

        # Restore original value
        array[index] = original_value

        # Numerical derivative
        numerical_gradient[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return numerical_gradient

def test_dense_full_gradient_check():

    rng = np.random.default_rng(42)

    X = rng.standard_normal((4, 2))

    layer = Dense(2, 3)

    layer.weights = rng.standard_normal((2, 3))
    layer.biases = rng.standard_normal((1, 3))

    epsilon = 1e-5

    # Analytical gradients
    Z = layer.forward(X)
    grad_output = Z

    grad_input = layer.backward(grad_output)

    analytical_grad_weights = layer.grad_weights.copy()
    analytical_grad_biases = layer.grad_biases.copy()
    analytical_grad_input = grad_input.copy()

    # Loss function
    def loss_fn():
        Z = layer.forward(X)
        return 0.5 * np.sum(Z ** 2)

    numerical_grad_weights = finite_difference_gradient(
        layer.weights, loss_fn, epsilon
    )

    numerical_grad_biases = finite_difference_gradient(
        layer.biases, loss_fn, epsilon
    )

    numerical_grad_input = finite_difference_gradient(
        X, loss_fn, epsilon
    )

    np.testing.assert_allclose(
        analytical_grad_weights,
        numerical_grad_weights,
        rtol=1e-5,
        atol=1e-7
    )

    np.testing.assert_allclose(
        analytical_grad_biases,
        numerical_grad_biases,
        rtol=1e-5,
        atol=1e-7
    )

    np.testing.assert_allclose(
        analytical_grad_input,
        numerical_grad_input,
        rtol=1e-5,
        atol=1e-7
    )

def test_sigmoid_gradient_check():

    X = np.array([
        [-2.0, 0.5],
        [1.0, 3.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    sigmoid = Sigmoid()

    sigmoid.forward(X)

    analytical_gradient = sigmoid.backward(grad_output).copy()

    def loss_fn():
        output = sigmoid.forward(X)
        return np.sum(output * grad_output)

    numerical_gradient = finite_difference_gradient(
        X,
        loss_fn,
        epsilon=1e-5
    )

    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )

def test_relu_gradient_check():

    X = np.array([
        [-2.0, 0.5],
        [1.0, 3.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    relu = ReLU()

    # Analytical gradient
    relu.forward(X)

    analytical_gradient = relu.backward(grad_output).copy()

    # Loss function
    def loss_fn():
        output = relu.forward(X)
        return np.sum(output * grad_output)

    # Numerical gradient
    numerical_gradient = finite_difference_gradient(
        X,
        loss_fn,
        epsilon=1e-5
    )

    # Gradient verification
    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )

def test_tanh_gradient_check():

    X = np.array([
        [-2.0, 0.5],
        [1.0, 3.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    tanh = Tanh()

    # 1. Analytical gradient
    tanh.forward(X)

    analytical_gradient = tanh.backward(grad_output).copy()

    # 2. Loss function
    def loss_fn():
        output = tanh.forward(X)
        return np.sum(output * grad_output)

    # 3. Numerical gradient
    numerical_gradient = finite_difference_gradient(
        X,
        loss_fn,
        epsilon=1e-5
    )

    # 4. Gradient verification
    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )

def test_softmax_gradient_check():

    X = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 2.0]
    ])

    grad_output = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    softmax = Softmax()

    # 1. Analytical gradient
    softmax.forward(X)

    analytical_gradient = softmax.backward(grad_output).copy()

    # 2. Loss function
    def loss_fn():
        output = softmax.forward(X)
        return np.sum(output * grad_output)

    # 3. Numerical gradient
    numerical_gradient = finite_difference_gradient(
        X,
        loss_fn,
        epsilon=1e-5
    )

    # 4. Gradient verification
    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )

    # 5. Verify that row gradients sum to zero
    np.testing.assert_allclose(
        np.sum(analytical_gradient, axis=1),
        np.zeros(X.shape[0]),
        atol=1e-7
    )

def test_mse_gradient_check():
    y_pred = np.array([
        [2.0, 4.0],
        [1.5, 3.0]
    ])

    y_true = np.array([
        [1.0, 2.0],
        [2.0, 2.5]
    ])

    mse = MSE()

    # Analytical gradient
    mse.forward(y_pred, y_true)
    analytical_gradient = mse.backward().copy()

    # Numerical gradient
    def loss_fn():
        return mse.forward(y_pred, y_true)

    numerical_gradient = finite_difference_gradient(
        y_pred,
        loss_fn,
        epsilon=1e-5
    )

    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )

def test_categorical_cross_entropy_gradient_check():

    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5]
    ])

    y_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    loss = CategoricalCrossEntropy()

    # 1. Analytical gradient
    loss.forward(y_pred, y_true)

    analytical_gradient = loss.backward().copy()

    # 2. Numerical loss function
    def loss_fn():
        return loss.forward(y_pred, y_true)

    # 3. Numerical gradient
    numerical_gradient = finite_difference_gradient(
        y_pred,
        loss_fn,
        epsilon=1e-5
    )

    # 4. Gradient verification
    np.testing.assert_allclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-7
    )