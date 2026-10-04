"""Command-line interface and entry point for polynomial interpolation project."""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

from interpolation.functions import (
    chebyshev_nodes,
    equally_spaced_nodes,
    runge_derivative,
    runge_function,
)
from interpolation.hermite import HermiteInterpolator
from interpolation.lagrange import LagrangeInterpolator
from interpolation.newton import NewtonInterpolator
from interpolation.plotting import InterpolationPlotter
from interpolation.runge_demo import RungeExperiment


def main() -> None:
    """CLI entry point for running interpolation demos and individual algorithms."""
    parser = argparse.ArgumentParser(
        description="Modular Polynomial Interpolation & Runge Phenomenon CLI"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="runge",
        choices=["runge", "none"],
        help="Run comprehensive Runge phenomenon experiment demo (default: runge).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["lagrange", "newton", "hermite", "all"],
        help="Algorithm to execute independently.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[5, 9, 13, 17, 21],
        help="Node counts sequence for Runge experiment.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Display Matplotlib figures interactively."
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save generated figures to PNG files."
    )

    args = parser.parse_args()

    if args.demo == "runge":
        exp = RungeExperiment(node_counts=args.nodes)
        exp.print_full_report()

        if args.save_plots or args.plot:
            prefix = "runge" if args.save_plots else None
            exp.generate_all_plots(save_prefix=prefix)
            if args.plot:
                plt.show()

    if args.algorithm != "all" and args.demo == "none":
        print(f"\n--- Running Standalone Algorithm Test: {args.algorithm.upper()} ---")
        x_nodes = equally_spaced_nodes(-1.0, 1.0, 7)
        y_nodes = runge_function(x_nodes)
        plotter = InterpolationPlotter()

        if args.algorithm == "lagrange":
            model = LagrangeInterpolator(x_nodes, y_nodes)
        elif args.algorithm == "newton":
            model = NewtonInterpolator(x_nodes, y_nodes)
        elif args.algorithm == "hermite":
            dy_nodes = runge_derivative(x_nodes)
            model = HermiteInterpolator(x_nodes, y_nodes, dy_nodes)

        plotter.plot_interpolation(
            runge_function, model, title=f"{args.algorithm.capitalize()} Interpolation"
        )
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
