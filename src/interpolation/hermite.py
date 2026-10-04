"""Hermite polynomial interpolation implementation."""

from typing import Self, Union
import numpy as np
import numpy.typing as npt

from interpolation.base import ArrayLike, BaseInterpolator


class HermiteInterpolator(BaseInterpolator):
    """Hermite polynomial interpolator matching function and derivative values."""

    def __init__(
        self,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        dydx: ArrayLike | None = None,
    ) -> None:
        self._dydx_nodes: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._z_nodes: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._coefficients: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._dd_table: npt.NDArray[np.float64] = np.array([[]], dtype=np.float64)

        if x is not None and y is not None and dydx is not None:
            self.fit(x, y, dydx)

    @property
    def derivative_nodes(self) -> npt.NDArray[np.float64]:
        """Return derivative values at nodes."""
        return self._dydx_nodes.copy()

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Return Hermite-Newton divided difference coefficients."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")
        return self._coefficients.copy()

    @property
    def degree(self) -> int:
        """Return polynomial degree (2N - 1 for N distinct nodes)."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")
        return len(self._z_nodes) - 1

    def fit(self, x: ArrayLike, y: ArrayLike, dydx: ArrayLike | None = None) -> Self:
        """Fit Hermite interpolating polynomial with extended divided differences table."""
        if dydx is None:
            raise ValueError("Hermite interpolation requires derivative array 'dydx'.")

        x_arr, y_arr = self._validate_nodes(x, y, allow_duplicates=False)

        try:
            dy_arr = np.asarray(dydx, dtype=np.float64).flatten()
        except (ValueError, TypeError) as err:
            raise TypeError("Derivative values must be numeric sequence.") from err

        if dy_arr.shape != x_arr.shape:
            raise ValueError("Derivative array 'dydx' shape must match 'x' shape.")

        if not np.all(np.isfinite(dy_arr)):
            raise ValueError("Derivative array must contain finite real numbers.")

        self._x_nodes = x_arr
        self._y_nodes = y_arr
        self._dydx_nodes = dy_arr

        n = len(x_arr)
        m = 2 * n

        z = np.zeros(m, dtype=np.float64)
        table = np.zeros((m, m), dtype=np.float64)

        for i in range(n):
            z[2 * i] = x_arr[i]
            z[2 * i + 1] = x_arr[i]
            table[2 * i, 0] = y_arr[i]
            table[2 * i + 1, 0] = y_arr[i]

        for i in range(m - 1):
            if i % 2 == 0:
                table[i, 1] = dy_arr[i // 2]
            else:
                denom = z[i + 1] - z[i]
                table[i, 1] = (table[i + 1, 0] - table[i, 0]) / denom

        for j in range(2, m):
            for i in range(m - j):
                denom = z[i + j] - z[i]
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / denom

        self._z_nodes = z
        self._dd_table = table
        self._coefficients = table[0, :].copy()
        self._is_fitted = True
        return self

    def evaluate(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate Hermite interpolating polynomial via Horner's scheme over z nodes."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        m = len(self._coefficients)
        result = np.full_like(x_flat, self._coefficients[-1], dtype=np.float64)

        for k in range(m - 2, -1, -1):
            result = result * (x_flat - self._z_nodes[k]) + self._coefficients[k]

        if is_scalar:
            return float(result[0])
        return result.reshape(x_input.shape)

    def derivative(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate derivative P'(x) of Hermite polynomial."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        m = len(self._coefficients)
        p_val = np.full_like(x_flat, self._coefficients[-1], dtype=np.float64)
        dp_val = np.zeros_like(x_flat, dtype=np.float64)

        for k in range(m - 2, -1, -1):
            dp_val = dp_val * (x_flat - self._z_nodes[k]) + p_val
            p_val = p_val * (x_flat - self._z_nodes[k]) + self._coefficients[k]

        if is_scalar:
            return float(dp_val[0])
        return dp_val.reshape(x_input.shape)
