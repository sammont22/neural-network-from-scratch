import numpy as np

from neural_network.activations import ReLU, Sigmoid, Tanh, Softmax

def test_relu_forward():

    X = np.array([
        [-2.0, 3.0],
        [4.0, -1.0]
    ])

    relu = ReLU()

    output = relu.forward(X)

    expected_output = np.array([
        [0.0, 3.0],
        [4.0, 0.0]
    ])

    np.testing.assert_allclose(
        output,
        expected_output
    )

def test_relu_backward():

    X = np.array([
        [-2.0, 3.0],
        [4.0, -1.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    relu = ReLU()

    relu.forward(X)

    grad_input = relu.backward(grad_output)

    expected_grad_input = np.array([
        [0.0, 2.0],
        [3.0, 0.0]
    ])

    np.testing.assert_allclose(
        grad_input,
        expected_grad_input
    )

def test_sigmoid_forward():

    X = np.array([
        [0.0, np.log(3)],
        [-np.log(3), 0.0]
    ])

    sigmoid = Sigmoid()

    output = sigmoid.forward(X)

    expected_output = np.array([
        [0.5, 0.75],
        [0.25, 0.5]
    ])

    np.testing.assert_allclose(
        output,
        expected_output
    )

def test_sigmoid_backward():
    X = np.array([
        [0.0, np.log(3)],
        [-np.log(3), 0.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    sigmoid = Sigmoid()

    sigmoid.forward(X)

    grad_input = sigmoid.backward(grad_output)

    expected_grad_input = np.array([
        [0.25, 0.375],
        [0.5625, 1.0]
    ])

    np.testing.assert_allclose(
        grad_input,
        expected_grad_input,
        rtol=1e-6,
        atol=1e-7
    )

def test_tanh_backward():
    X = np.array([
        [0.0, 1.0],
        [-1.0, 0.0]
    ])

    grad_output = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    tanh = Tanh()

    tanh.forward(X)

    grad_input = tanh.backward(grad_output)

    expected_grad_input = np.array([
        [1.0, 0.83994868],
        [1.25992302, 4.0]
    ])

    np.testing.assert_allclose(
        grad_input,
        expected_grad_input,
        rtol=1e-6,
        atol=1e-7
    )

def test_softmax_forward():

    X = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ])

    softmax = Softmax()

    output = softmax.forward(X)

    expected_output = np.array([
        [0.09003057, 0.24472847, 0.66524096],
        [1/3, 1/3, 1/3]
    ])

    np.testing.assert_allclose(
        output,
        expected_output,
        rtol=1e-6,
        atol=1e-7
    )

    np.testing.assert_allclose(
        np.sum(output, axis=1),
        np.ones(X.shape[0])
    )

def test_softmax_backward():
    X = np.array([
        [0.0, 0.0, 0.0]
    ])

    grad_output = np.array([
        [1.0, 2.0, 3.0]
    ])

    softmax = Softmax()

    # Forward propagation
    softmax.forward(X)

    # Backward propagation
    grad_input = softmax.backward(grad_output)

    # Expected gradients
    expected_grad_input = np.array([
        [-1/3, 0.0, 1/3]
    ])

    # Numerical verification
    np.testing.assert_allclose(
        grad_input,
        expected_grad_input,
        rtol=1e-6,
        atol=1e-7
    )