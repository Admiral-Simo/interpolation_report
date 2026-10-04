"""Unit tests for HermiteInterpolator."""

import numpy as np
import pytest

from interpolation.hermite import HermiteInterpolator


def test_hermite_matches_values_and_derivatives():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([1.0, 3.0, 2.0])
    dy_nodes = np.array([0.0, -1.0, 2.0])

    interp = HermiteInterpolator(x_nodes, y_nodes, dy_nodes)

    for x_i, y_i, dy_i in zip(x_nodes, y_nodes, dy_nodes):
        assert np.isclose(interp(x_i), y_i, atol=1e-12)
        assert np.isclose(interp.derivative(x_i), dy_i, atol=1e-12)


def test_hermite_degree():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([1.0, 2.0, 5.0])
    dy_nodes = np.array([1.0, 2.0, 4.0])
    interp = HermiteInterpolator(x_nodes, y_nodes, dy_nodes)

    # 3 unique nodes with derivatives => 6 conditions => degree 5
    assert interp.degree == 5


def test_hermite_invalid_input():
    with pytest.raises(ValueError, match="Derivative array 'dydx' shape must match"):
        HermiteInterpolator([0.0, 1.0], [0.0, 1.0], [1.0])
