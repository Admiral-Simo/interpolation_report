"""Unit tests for NewtonInterpolator."""

import numpy as np
import pytest

from interpolation.newton import NewtonInterpolator


def test_newton_passes_through_nodes():
    x_nodes = np.array([-1.0, 0.0, 2.0])
    y_nodes = np.array([4.0, 1.0, 7.0])
    interp = NewtonInterpolator(x_nodes, y_nodes)

    for x_i, y_i in zip(x_nodes, y_nodes):
        assert np.isclose(interp(x_i), y_i, atol=1e-12)


def test_newton_divided_differences():
    # f(x) = x^2 => f[0]=0, f[1]=1, f[2]=4
    # f[0,1] = (1-0)/1 = 1; f[1,2] = (4-1)/1 = 3
    # f[0,1,2] = (3-1)/(2-0) = 1
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([0.0, 1.0, 4.0])
    interp = NewtonInterpolator(x_nodes, y_nodes)

    coeffs = interp.coefficients
    assert np.allclose(coeffs, [0.0, 1.0, 1.0])


def test_newton_mismatched_length_rejection():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        NewtonInterpolator([1.0, 2.0], [1.0])
