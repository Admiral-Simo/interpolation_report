"""Visualization component for polynomial interpolation analysis."""

from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np

from interpolation.base import BaseInterpolator


class InterpolationPlotter:
    """Dedicated plotting renderer for interpolation models and error analysis."""

    def __init__(self, figsize: tuple[int, int] = (10, 6), style: str = "seaborn-v0_8-whitegrid") -> None:
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")

    def plot_interpolation(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        interpolator: BaseInterpolator,
        interval: tuple[float, float] = (-1.0, 1.0),
        num_grid_points: int = 1000,
        title: str = "Polynomial Interpolation",
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot true function, interpolation nodes, and interpolating polynomial."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=120)

        x_grid = np.linspace(interval[0], interval[1], num_grid_points)
        y_true = function(x_grid)
        y_pred = interpolator(x_grid)

        x_nodes = interpolator.x_nodes
        y_nodes = interpolator.y_nodes

        ax.plot(x_grid, y_true, "k-", linewidth=2.0, label="Exact Function f(x)")
        ax.plot(x_grid, y_pred, "r--", linewidth=1.8, label=f"Interpolant P(x) (deg={interpolator.degree})")
        ax.plot(x_nodes, y_nodes, "ko", markersize=7, markerfacecolor="orange", markeredgewidth=1.5, label="Nodes")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(frameon=True, loc="best", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_runge_comparison(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        node_counts: Sequence[int],
        interpolator_cls: type[BaseInterpolator],
        interval: tuple[float, float] = (-1.0, 1.0),
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot Runge phenomenon progression across node counts for equal vs Chebyshev nodes."""
        from interpolation.functions import chebyshev_nodes, equally_spaced_nodes

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)
        x_grid = np.linspace(interval[0], interval[1], 1000)
        y_true = function(x_grid)

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(node_counts)))

        axes[0].plot(x_grid, y_true, "k-", linewidth=2.5, label="Runge f(x)")
        axes[1].plot(x_grid, y_true, "k-", linewidth=2.5, label="Runge f(x)")

        for idx, n in enumerate(node_counts):
            x_eq = equally_spaced_nodes(interval[0], interval[1], n)
            y_eq = function(x_eq)
            interp_eq = interpolator_cls(x_eq, y_eq)
            axes[0].plot(x_grid, interp_eq(x_grid), color=colors[idx], label=f"N={n}")

            x_ch = chebyshev_nodes(interval[0], interval[1], n)
            y_ch = function(x_ch)
            interp_ch = interpolator_cls(x_ch, y_ch)
            axes[1].plot(x_grid, interp_ch(x_grid), color=colors[idx], label=f"N={n}")

        axes[0].set_title("Equally Spaced Nodes (Runge Phenomenon)", fontsize=12, fontweight="bold")
        axes[1].set_title("Chebyshev Nodes (Suppressed Oscillations)", fontsize=12, fontweight="bold")

        for ax in axes:
            ax.set_ylim(-0.5, 2.0)
            ax.set_xlabel("x", fontsize=11)
            ax.legend(frameon=True, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

        axes[0].set_ylabel("y", fontsize=11)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_error_curves(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        node_counts: Sequence[int],
        interpolator_cls: type[BaseInterpolator],
        interval: tuple[float, float] = (-1.0, 1.0),
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot logarithmic interpolation error profiles |f(x) - P(x)|."""
        from interpolation.functions import chebyshev_nodes, equally_spaced_nodes

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
        x_grid = np.linspace(interval[0], interval[1], 1000)
        y_true = function(x_grid)

        for n in node_counts:
            x_eq = equally_spaced_nodes(interval[0], interval[1], n)
            y_eq = function(x_eq)
            interp_eq = interpolator_cls(x_eq, y_eq)
            err_eq = np.abs(y_true - interp_eq(x_grid))
            axes[0].semilogy(x_grid, err_eq + 1e-16, label=f"N={n}")

            x_ch = chebyshev_nodes(interval[0], interval[1], n)
            y_ch = function(x_ch)
            interp_ch = interpolator_cls(x_ch, y_ch)
            err_ch = np.abs(y_true - interp_ch(x_grid))
            axes[1].semilogy(x_grid, err_ch + 1e-16, label=f"N={n}")

        axes[0].set_title("Error: Equally Spaced Nodes", fontsize=12, fontweight="bold")
        axes[1].set_title("Error: Chebyshev Nodes", fontsize=12, fontweight="bold")

        for ax in axes:
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel("|f(x) - P(x)|", fontsize=11)
            ax.legend(frameon=True, fontsize=9)
            ax.grid(True, which="both", linestyle=":", alpha=0.6)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_max_error_vs_nodes(
        self,
        node_counts: Sequence[int],
        eq_max_errors: Sequence[float],
        ch_max_errors: Sequence[float],
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot Max Error vs Number of Nodes on a log-scale."""
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)

        ax.semilogy(node_counts, eq_max_errors, "ro-", linewidth=2, label="Equally Spaced Nodes")
        ax.semilogy(node_counts, ch_max_errors, "bs--", linewidth=2, label="Chebyshev Nodes")

        ax.set_title("Maximum Interpolation Error Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Interpolation Nodes (N)", fontsize=11)
        ax.set_ylabel("Maximum Absolute Error (log scale)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
