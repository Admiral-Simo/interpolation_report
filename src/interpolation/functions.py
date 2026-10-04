"""Mathematical test functions, node generators, and numerical error metrics."""

from typing import Callable, Union
import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, int, list[float], npt.NDArray[np.float64]]


def runge_function(x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
    """Compute classical Runge function f(x) = 1 / (1 + 25 * x^2)."""
    x_arr = np.asarray(x, dtype=np.float64)
    res = 1.0 / (1.0 + 25.0 * (x_arr**2))
    if x_arr.ndim == 0:
        return float(res)
    return res


def runge_derivative(x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
    """Compute derivative of Runge function f'(x) = -50 * x / (1 + 25 * x^2)^2."""
    x_arr = np.asarray(x, dtype=np.float64)
    res = -50.0 * x_arr / ((1.0 + 25.0 * (x_arr**2)) ** 2)
    if x_arr.ndim == 0:
        return float(res)
    return res


def equally_spaced_nodes(a: float, b: float, n: int) -> npt.NDArray[np.float64]:
    """Generate n equally spaced nodes on interval [a, b]."""
    if n < 2:
        raise ValueError("Number of nodes n must be at least 2.")
    if a >= b:
        raise ValueError(f"Interval bounds must satisfy a < b. Got [{a}, {b}].")
    return np.linspace(a, b, n, dtype=np.float64)


def chebyshev_nodes(a: float, b: float, n: int) -> npt.NDArray[np.float64]:
    """Generate n Chebyshev nodes of the first kind transformed to interval [a, b]."""
    if n < 1:
        raise ValueError("Number of nodes n must be at least 1.")
    if a >= b:
        raise ValueError(f"Interval bounds must satisfy a < b. Got [{a}, {b}].")

    indices = np.arange(1, n + 1, dtype=np.float64)
    x_cheb = np.cos((2.0 * indices - 1.0) * np.pi / (2.0 * n))
    x_scaled = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb
    return np.sort(x_scaled)


def max_absolute_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute Maximum Absolute Error (L_infinity norm)."""
    return float(np.max(np.abs(y_true - y_pred)))


def rmse(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], eps: float = 1e-15
) -> float:
    """Compute Maximum Relative Error."""
    max_err = max_absolute_error(y_true, y_pred)
    denom = float(np.max(np.abs(y_true)))
    return max_err / (denom + eps)
