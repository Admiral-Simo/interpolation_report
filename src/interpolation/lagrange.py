"""Lagrange polynomial interpolation implementation."""

from typing import Self, Union
import numpy as np
import numpy.typing as npt

from interpolation.base import ArrayLike, BaseInterpolator


class LagrangeInterpolator(BaseInterpolator):
    """Lagrange polynomial interpolator utilizing Second Barycentric Form.

    Complexity
    ----------
    Precomputation (fitting) : O(N^2) time to compute barycentric weights.
    Evaluation               : O(N) time per evaluation point.
    """

    def __init__(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> None:
        self._weights: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        super().__init__(x=x, y=y)

    def fit(self, x: ArrayLike, y: ArrayLike) -> Self:
        """Fit Lagrange interpolator by computing barycentric weights."""
        x_arr, y_arr = self._validate_nodes(x, y, allow_duplicates=False)
        self._x_nodes = x_arr
        self._y_nodes = y_arr

        n = len(x_arr)
        weights = np.ones(n, dtype=np.float64)
        for i in range(n):
            diffs = x_arr[i] - x_arr[np.arange(n) != i]
            weights[i] = 1.0 / np.prod(diffs)

        self._weights = weights
        self._is_fitted = True
        return self

    def basis_polynomial(self, i: int, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate i-th Lagrange basis polynomial L_i(x) directly."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")
        if i < 0 or i >= len(self._x_nodes):
            raise IndexError(f"Basis index {i} out of bounds for {len(self._x_nodes)} nodes.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        n = len(self._x_nodes)
        x_i = self._x_nodes[i]
        denom = np.prod(x_i - self._x_nodes[np.arange(n) != i])

        result = np.ones_like(x_flat, dtype=np.float64)
        for j in range(n):
            if j != i:
                result *= (x_flat - self._x_nodes[j])
        result /= denom

        if is_scalar:
            return float(result[0])
        return result.reshape(x_input.shape)

    def evaluate(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate interpolating polynomial via Barycentric Formula."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        diffs = x_flat[:, np.newaxis] - self._x_nodes[np.newaxis, :]
        exact_matches = np.isclose(diffs, 0.0, atol=1e-14)

        result = np.zeros(x_flat.shape, dtype=np.float64)

        for idx in range(len(x_flat)):
            match_indices = np.where(exact_matches[idx])[0]
            if len(match_indices) > 0:
                result[idx] = self._y_nodes[match_indices[0]]
            else:
                terms = self._weights / diffs[idx]
                result[idx] = np.sum(terms * self._y_nodes) / np.sum(terms)

        if is_scalar:
            return float(result[0])
        return result.reshape(x_input.shape)

    def derivative(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate derivative P'(x) using Lagrange basis derivatives."""
        if not self._is_fitted:
            raise ValueError("Interpolator is not fitted.")

        x_input = np.asarray(x, dtype=np.float64)
        is_scalar = x_input.ndim == 0
        x_flat = np.atleast_1d(x_input)

        n = len(self._x_nodes)
        derivatives = np.zeros_like(x_flat, dtype=np.float64)

        for i in range(n):
            dL_i = np.zeros_like(x_flat, dtype=np.float64)
            for j in range(n):
                if j == i:
                    continue
                term = np.ones_like(x_flat, dtype=np.float64)
                for k in range(n):
                    if k != i and k != j:
                        term *= (x_flat - self._x_nodes[k]) / (self._x_nodes[i] - self._x_nodes[k])
                term /= (self._x_nodes[i] - self._x_nodes[j])
                dL_i += term
            derivatives += self._y_nodes[i] * dL_i

        if is_scalar:
            return float(derivatives[0])
        return derivatives.reshape(x_input.shape)
