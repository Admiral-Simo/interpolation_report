"""Newton polynomial interpolation implementation."""

from typing import Self, Union
import numpy as np
import numpy.typing as npt

from interpolation.base import ArrayLike, BaseInterpolator


class NewtonInterpolator(BaseInterpolator):
    """Newton polynomial interpolator using divided differences and Horner's evaluation."""

    def __init__(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> None:
        self._coefficients: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._dd_table: npt.NDArray[np.float64] = np.array([[]], dtype=np.float64)
        super().__init__(x=x, y=y)

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Return Newton basis coefficients f[x0, ..., xk]."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")
        return self._coefficients.copy()

    @property
    def divided_difference_table(self) -> npt.NDArray[np.float64]:
        """Return complete divided difference table matrix."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")
        return self._dd_table.copy()

    def fit(self, x: ArrayLike, y: ArrayLike) -> Self:
        """Compute divided difference table and store Newton coefficients."""
        x_arr, y_arr = self._validate_nodes(x, y, allow_duplicates=False)
        self._x_nodes = x_arr
        self._y_nodes = y_arr

        n = len(x_arr)
        table = np.zeros((n, n), dtype=np.float64)
        table[:, 0] = y_arr

        for j in range(1, n):
            for i in range(n - j):
                denom = x_arr[i + j] - x_arr[i]
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / denom

        self._dd_table = table
        self._coefficients = table[0, :].copy()
        self._is_fitted = True
        return self

    def evaluate(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate interpolating polynomial via Horner's nested multiplication."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        n = len(self._coefficients)
        result = np.full_like(x_flat, self._coefficients[-1], dtype=np.float64)

        for k in range(n - 2, -1, -1):
            result = result * (x_flat - self._x_nodes[k]) + self._coefficients[k]

        if is_scalar:
            return float(result[0])
        return result.reshape(x_input.shape)

    def derivative(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate polynomial derivative P'(x) using Horner differentiation scheme."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        n = len(self._coefficients)
        p_val = np.full_like(x_flat, self._coefficients[-1], dtype=np.float64)
        dp_val = np.zeros_like(x_flat, dtype=np.float64)

        for k in range(n - 2, -1, -1):
            dp_val = dp_val * (x_flat - self._x_nodes[k]) + p_val
            p_val = p_val * (x_flat - self._x_nodes[k]) + self._coefficients[k]

        if is_scalar:
            return float(dp_val[0])
        return dp_val.reshape(x_input.shape)
