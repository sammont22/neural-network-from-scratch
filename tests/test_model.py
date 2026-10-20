import numpy as np

from neural_network.model import NeuralNetwork
from neural_network.layers import Dense
from neural_network.activations import ReLU
from neural_network.optimizers import SGD
from neural_network.losses import MSE

def test_neural_network_shapes():

    model = NeuralNetwork([
        Dense(2, 4),
        ReLU(),
        Dense(4, 1)
    ])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    # Forward
    output = model.forward(X)

    assert output.shape == (2, 1)

    # Backward
    grad_output = np.ones_like(output)

    grad_input = model.backward(grad_output)

    assert grad_input.shape == X.shape

def test_neural_network_step():

    dense1 = Dense(2, 3)
    relu = ReLU()
    dense2 = Dense(3, 1)

    model = NeuralNetwork([
        dense1,
        relu,
        dense2
    ])

    # Assign known gradients
    for layer in [dense1, dense2]:
        layer.grad_weights = np.ones_like(layer.weights)
        layer.grad_biases = np.ones_like(layer.biases)

    # Store original parameters
    original_weights = [
        dense1.weights.copy(),
        dense2.weights.copy()
    ]

    original_biases = [
        dense1.biases.copy(),
        dense2.biases.copy()
    ]

    # Update all trainable layers
    optimizer = SGD(learning_rate=0.01)
    model.step(optimizer)

    # Verify updates
    for i, layer in enumerate([dense1, dense2]):

        expected_weights = original_weights[i] - 0.01
        expected_biases = original_biases[i] - 0.01

        np.testing.assert_allclose(
            layer.weights,
            expected_weights
        )

        np.testing.assert_allclose(
            layer.biases,
            expected_biases
        )

def test_neural_network_fit():

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    y_true = np.array([
        [1.0],
        [2.0]
    ])

    dense = Dense(2, 1)

    dense.weights = np.array([
        [0.1],
        [0.2]
    ])

    dense.biases = np.array([[0.0]])

    model = NeuralNetwork([dense])

    loss_fn = MSE()
    optimizer = SGD(learning_rate=0.01)

    # Training
    history = model.fit(
        X,
        y_true,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=100
    )

    # Evaluate final loss
    final_predictions = model.forward(X)
    final_loss = loss_fn.forward(final_predictions, y_true)

    # Verify training
    assert len(history) == 100
    assert final_loss < history[0]
    assert final_predictions.shape == y_true.shape

def test_neural_network_predict():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    dense = Dense(2, 1)

    dense.weights = np.array([
        [0.1],
        [0.2]
    ])

    dense.biases = np.array([[0.0]])

    model = NeuralNetwork([dense])

    predictions = model.predict(X)

    expected_predictions = np.array([
        [0.5],
        [1.1]
    ])

    np.testing.assert_allclose(
        predictions,
        expected_predictions
    )

def test_neural_network_multilayer_fit():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    y_true = np.array([
        [1.0],
        [2.0]
    ])

    dense1 = Dense(2, 4)
    dense2 = Dense(4, 1)

    # Deterministic initialization
    dense1.weights = np.array([
        [0.1, 0.2, 0.1, 0.2],
        [0.2, 0.1, 0.2, 0.1]
    ])
    dense1.biases = np.zeros((1, 4))

    dense2.weights = np.array([
        [0.1],
        [0.2],
        [0.1],
        [0.2]
    ])
    dense2.biases = np.zeros((1, 1))

    model = NeuralNetwork([
        dense1,
        ReLU(),
        dense2
    ])

    loss_fn = MSE()
    optimizer = SGD(learning_rate=0.01)

    history = model.fit(
        X,
        y_true,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=100
    )

    predictions = model.predict(X)
    final_loss = loss_fn.forward(predictions, y_true)

    assert len(history) == 100
    assert final_loss < history[0]
    assert predictions.shape == y_true.shape