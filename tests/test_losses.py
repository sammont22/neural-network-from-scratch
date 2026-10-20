import numpy as np

from neural_network.losses import MSE, CategoricalCrossEntropy


def test_mse_forward():

    y_pred = np.array([
        [2.0, 4.0]
    ])

    y_true = np.array([
        [1.0, 2.0]
    ])

    mse = MSE()

    loss = mse.forward(y_pred, y_true)

    expected_loss = 2.5

    assert np.isclose(loss, expected_loss)

def test_mse_backward():
    y_pred = np.array([
        [2.0, 4.0]
    ])

    y_true = np.array([
        [1.0, 2.0]
    ])

    mse = MSE()

    mse.forward(y_pred, y_true)

    grad_pred = mse.backward()

    expected_gradient = np.array([
        [1.0, 2.0]
    ])

    np.testing.assert_allclose(
        grad_pred,
        expected_gradient,
        rtol=1e-6,
        atol=1e-7
    )

def test_categorical_cross_entropy_forward():
    y_pred = np.array([
        [0.8, 0.2],
        [0.25, 0.75]
    ])

    y_true = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    loss_fn = CategoricalCrossEntropy()

    loss = loss_fn.forward(y_pred, y_true)

    expected_loss = 0.2554128119

    assert np.isclose(loss, expected_loss, atol=1e-8)

def test_categorical_cross_entropy_backward():
    y_pred = np.array([
        [0.8, 0.2],
        [0.25, 0.75]
    ])

    y_true = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    loss_fn = CategoricalCrossEntropy()

    loss_fn.forward(y_pred, y_true)

    grad_pred = loss_fn.backward()

    expected_gradient = np.array([
        [-0.625, 0.0],
        [0.0, -2.0 / 3.0]
    ])

    np.testing.assert_allclose(
        grad_pred,
        expected_gradient,
        rtol=1e-6,
        atol=1e-7
    )