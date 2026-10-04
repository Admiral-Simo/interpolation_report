"""Cross-algorithm consistency and polynomial reconstruction tests."""

import numpy as np

from interpolation.functions import runge_function
from interpolation.lagrange import LagrangeInterpolator
from interpolation.newton import NewtonInterpolator


def test_exact_polynomial_reconstruction():
    # f(x) = 2x^3 - 3x^2 + x + 5
    poly = lambda x: 2.0 * x**3 - 3.0 * x**2 + x + 5.0

    x_nodes = np.array([-1.0, 0.5, 1.5, 3.0])
    y_nodes = poly(x_nodes)

    lagrange = LagrangeInterpolator(x_nodes, y_nodes)
    newton = NewtonInterpolator(x_nodes, y_nodes)

    test_grid = np.linspace(-2.0, 4.0, 100)
    y_true = poly(test_grid)

    assert np.allclose(lagrange(test_grid), y_true, atol=1e-12)
    assert np.allclose(newton(test_grid), y_true, atol=1e-12)


def test_lagrange_newton_equivalence_on_runge():
    x_nodes = np.linspace(-1.0, 1.0, 13)
    y_nodes = runge_function(x_nodes)

    lagrange = LagrangeInterpolator(x_nodes, y_nodes)
    newton = NewtonInterpolator(x_nodes, y_nodes)

    x_grid = np.linspace(-1.0, 1.0, 500)
    diff = np.max(np.abs(lagrange(x_grid) - newton(x_grid)))

    assert diff < 1e-13
