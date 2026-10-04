"""Unit tests for LagrangeInterpolator."""

import numpy as np
import pytest

from interpolation.lagrange import LagrangeInterpolator


def test_lagrange_passes_through_nodes():
    x_nodes = np.array([-2.0, 0.0, 1.0, 3.0])
    y_nodes = np.array([5.0, 1.0, 2.0, 10.0])
    interp = LagrangeInterpolator(x_nodes, y_nodes)

    for x_i, y_i in zip(x_nodes, y_nodes):
        assert np.isclose(interp(x_i), y_i, atol=1e-12)


def test_lagrange_basis_polynomial():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([0.0, 1.0, 4.0])
    interp = LagrangeInterpolator(x_nodes, y_nodes)

    # L_1(1.0) = 1.0, L_1(0.0) = 0.0
    assert np.isclose(interp.basis_polynomial(1, 1.0), 1.0)
    assert np.isclose(interp.basis_polynomial(1, 0.0), 0.0)


def test_lagrange_scalar_and_vector():
    x_nodes = np.array([1.0, 2.0, 3.0])
    y_nodes = np.array([2.0, 4.0, 8.0])
    interp = LagrangeInterpolator(x_nodes, y_nodes)

    val_scalar = interp(1.5)
    assert isinstance(val_scalar, float)

    val_array = interp(np.array([1.5, 2.5]))
    assert isinstance(val_array, np.ndarray)
    assert val_array.shape == (2,)


def test_lagrange_duplicate_node_rejection():
    x_nodes = np.array([1.0, 2.0, 2.0])
    y_nodes = np.array([1.0, 4.0, 4.0])
    with pytest.raises(ValueError, match="must be distinct"):
        LagrangeInterpolator(x_nodes, y_nodes)
