"""Base class and common interface for polynomial interpolators."""

from abc import ABC, abstractmethod
from typing import Self, Union
import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, int, list[float], npt.NDArray[np.float64]]


class BaseInterpolator(ABC):
    """Abstract base class for polynomial interpolators."""

    def __init__(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> None:
        self._x_nodes: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._y_nodes: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._is_fitted: bool = False

        if x is not None and y is not None:
            self.fit(x, y)

    @property
    def x_nodes(self) -> npt.NDArray[np.float64]:
        """Return a copy of interpolation x-nodes."""
        return self._x_nodes.copy()

    @property
    def y_nodes(self) -> npt.NDArray[np.float64]:
        """Return a copy of function values at interpolation nodes."""
        return self._y_nodes.copy()

    @property
    def degree(self) -> int:
        """Return degree of interpolating polynomial."""
        if not self._is_fitted:
            raise ValueError("Interpolator has not been fitted yet.")
        return len(self._x_nodes) - 1

    @staticmethod
    def _validate_nodes(
        x: ArrayLike, y: ArrayLike, allow_duplicates: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Validate input nodes x and values y."""
        try:
            x_arr = np.asarray(x, dtype=np.float64).flatten()
            y_arr = np.asarray(y, dtype=np.float64).flatten()
        except (ValueError, TypeError) as err:
            raise TypeError("Node inputs must be numeric arrays or sequences.") from err

        if x_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Node arrays cannot be empty.")

        if x_arr.shape != y_arr.shape:
            raise ValueError(
                f"Dimension mismatch: x shape {x_arr.shape} does not match y shape {y_arr.shape}."
            )

        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("Interpolation nodes and values must contain finite real numbers.")

        if not allow_duplicates and len(np.unique(np.round(x_arr, decimals=14))) < len(x_arr):
            raise ValueError("Interpolation nodes (x-values) must be distinct.")

        return x_arr, y_arr

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike) -> Self:
        """Fit the interpolating polynomial to data points (x, y)."""

    @abstractmethod
    def evaluate(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate interpolating polynomial at target x points."""

    @abstractmethod
    def derivative(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Evaluate first derivative of interpolating polynomial at target x points."""

    def __call__(self, x: ArrayLike) -> Union[float, npt.NDArray[np.float64]]:
        """Enable callable syntax for interpolator evaluation."""
        return self.evaluate(x)
