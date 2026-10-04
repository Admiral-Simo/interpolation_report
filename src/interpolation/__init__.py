"""Polynomial Interpolation Library."""

from interpolation.base import BaseInterpolator
from interpolation.functions import (
    chebyshev_nodes,
    equally_spaced_nodes,
    max_absolute_error,
    relative_error,
    rmse,
    runge_derivative,
    runge_function,
)
from interpolation.hermite import HermiteInterpolator
from interpolation.lagrange import LagrangeInterpolator
from interpolation.newton import NewtonInterpolator
from interpolation.plotting import InterpolationPlotter
from interpolation.runge_demo import RungeExperiment

__all__ = [
    "BaseInterpolator",
    "LagrangeInterpolator",
    "NewtonInterpolator",
    "HermiteInterpolator",
    "InterpolationPlotter",
    "RungeExperiment",
    "runge_function",
    "runge_derivative",
    "equally_spaced_nodes",
    "chebyshev_nodes",
    "max_absolute_error",
    "rmse",
    "relative_error",
]
