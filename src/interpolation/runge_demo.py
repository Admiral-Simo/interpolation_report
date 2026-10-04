"""Automated demonstration experiment and report generator for the Runge phenomenon."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from interpolation.functions import (
    chebyshev_nodes,
    equally_spaced_nodes,
    max_absolute_error,
    rmse,
    runge_derivative,
    runge_function,
)
from interpolation.hermite import HermiteInterpolator
from interpolation.lagrange import LagrangeInterpolator
from interpolation.newton import NewtonInterpolator
from interpolation.plotting import InterpolationPlotter


@dataclass
class ExperimentResult:
    """Dataclass holding error metrics for an interpolation configuration."""

    num_nodes: int
    node_type: str
    max_error: float
    rmse_error: float


class RungeExperiment:
    """Experiment runner analyzing Runge Phenomenon across algorithms and node types."""

    def __init__(
        self,
        node_counts: Sequence[int] = (5, 9, 13, 17, 21),
        interval: tuple[float, float] = (-1.0, 1.0),
        grid_resolution: int = 1000,
    ) -> None:
        self.node_counts = list(node_counts)
        self.interval = interval
        self.grid_resolution = grid_resolution
        self.x_grid = np.linspace(interval[0], interval[1], grid_resolution)
        self.y_true = runge_function(self.x_grid)
        self.plotter = InterpolationPlotter()

    def run_node_distribution_experiment(self) -> tuple[list[ExperimentResult], list[ExperimentResult]]:
        """Run convergence comparison between equally spaced and Chebyshev nodes."""
        eq_results: list[ExperimentResult] = []
        ch_results: list[ExperimentResult] = []

        for n in self.node_counts:
            x_eq = equally_spaced_nodes(self.interval[0], self.interval[1], n)
            y_eq = runge_function(x_eq)
            interp_eq = NewtonInterpolator(x_eq, y_eq)
            y_pred_eq = interp_eq(self.x_grid)
            eq_results.append(
                ExperimentResult(
                    num_nodes=n,
                    node_type="Equally spaced",
                    max_error=max_absolute_error(self.y_true, y_pred_eq),
                    rmse_error=rmse(self.y_true, y_pred_eq),
                )
            )

            x_ch = chebyshev_nodes(self.interval[0], self.interval[1], n)
            y_ch = runge_function(x_ch)
            interp_ch = NewtonInterpolator(x_ch, y_ch)
            y_pred_ch = interp_ch(self.x_grid)
            ch_results.append(
                ExperimentResult(
                    num_nodes=n,
                    node_type="Chebyshev",
                    max_error=max_absolute_error(self.y_true, y_pred_ch),
                    rmse_error=rmse(self.y_true, y_pred_ch),
                )
            )

        return eq_results, ch_results

    def compare_lagrange_vs_newton(self, n: int = 15) -> float:
        """Verify numerical equivalence between Lagrange and Newton interpolants."""
        x_nodes = equally_spaced_nodes(self.interval[0], self.interval[1], n)
        y_nodes = runge_function(x_nodes)

        lagrange = LagrangeInterpolator(x_nodes, y_nodes)
        newton = NewtonInterpolator(x_nodes, y_nodes)

        y_lagrange = lagrange(self.x_grid)
        y_newton = newton(self.x_grid)

        return max_absolute_error(y_lagrange, y_newton)

    def run_hermite_verification(
        self, nodes: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    ) -> list[tuple[float, float, float]]:
        """Verify Hermite interpolation matches function and derivative values at nodes."""
        x_nodes = np.array(nodes, dtype=np.float64)
        y_nodes = runge_function(x_nodes)
        dy_nodes = runge_derivative(x_nodes)

        hermite = HermiteInterpolator(x_nodes, y_nodes, dy_nodes)

        verifications = []
        for x_i, y_i, dy_i in zip(x_nodes, y_nodes, dy_nodes):
            val_err = abs(hermite(x_i) - y_i)
            der_err = abs(hermite.derivative(x_i) - dy_i)
            verifications.append((x_i, val_err, der_err))

        return verifications

    def print_full_report(self) -> None:
        """Execute and output complete textual demonstration report."""
        print("=" * 68)
        print("     POLYNOMIAL INTERPOLATION & RUNGE PHENOMENON DEMO")
        print("=" * 68)
        print(f"Function : Runge function f(x) = 1 / (1 + 25x^2)")
        print(f"Interval : [{self.interval[0]}, {self.interval[1]}]\n")

        eq_res, ch_res = self.run_node_distribution_experiment()

        print("-" * 68)
        print("Equally Spaced Nodes")
        print("-" * 68)
        print(f"{'Nodes (N)':<12}{'Max Error':<20}{'RMSE':<20}")
        print("-" * 68)
        for r in eq_res:
            print(f"{r.num_nodes:<12}{r.max_error:<20.6e}{r.rmse_error:<20.6e}")

        print("\n" + "-" * 68)
        print("Chebyshev Nodes")
        print("-" * 68)
        print(f"{'Nodes (N)':<12}{'Max Error':<20}{'RMSE':<20}")
        print("-" * 68)
        for r in ch_res:
            print(f"{r.num_nodes:<12}{r.max_error:<20.6e}{r.rmse_error:<20.6e}")

        diff = self.compare_lagrange_vs_newton(n=15)
        print("\n" + "-" * 68)
        print("Algorithm Equivalence Check (N=15)")
        print("-" * 68)
        print(f"Lagrange vs Newton maximum difference: {diff:.3e}")

        hermite_verif = self.run_hermite_verification()
        print("\n" + "-" * 68)
        print("Hermite Interpolation Verification")
        print("-" * 68)
        print(f"{'Node (x_i)':<15}{'Function Error':<25}{'Derivative Error':<25}")
        print("-" * 68)
        for x_i, val_err, der_err in hermite_verif:
            print(f"{x_i:<15.2f}{val_err:<25.3e}{der_err:<25.3e}")
        print("=" * 68 + "\n")

    def generate_all_plots(self, save_prefix: str | None = None) -> None:
        """Generate and display/save all 5 project visualization figures."""
        eq_res, ch_res = self.run_node_distribution_experiment()

        path1 = f"{save_prefix}_runge_curves.png" if save_prefix else None
        self.plotter.plot_runge_comparison(
            runge_function, self.node_counts, NewtonInterpolator, save_path=path1
        )

        path2 = f"{save_prefix}_error_curves.png" if save_prefix else None
        self.plotter.plot_error_curves(
            runge_function, self.node_counts, NewtonInterpolator, save_path=path2
        )

        path3 = f"{save_prefix}_convergence.png" if save_prefix else None
        eq_max = [r.max_error for r in eq_res]
        ch_max = [r.max_error for r in ch_res]
        self.plotter.plot_max_error_vs_nodes(
            self.node_counts, eq_max, ch_max, save_path=path3
        )
