import numpy as np

from neural_network.layers import Dense


def test_dense_forward_shape():

    X = np.random.default_rng(42).standard_normal((32, 5))

    layer = Dense(5, 10)

    Z = layer.forward(X)

    assert Z.shape == (32, 10)