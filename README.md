# Modular Polynomial Interpolation & Runge Phenomenon Analysis

A clean Python library implementing Lagrange, Newton, and Hermite polynomial interpolation algorithms, complete with error metrics, Matplotlib visualization tools, and a demonstration suite for analyzing the Runge phenomenon.

## Features
- **Lagrange Interpolation:** Implemented via Second Barycentric Form for $\mathcal{O}(N)$ evaluation complexity.
- **Newton Interpolation:** Constructs divided difference table with Horner's nested multiplication.
- **Hermite Interpolation:** Supports derivative matching via repeated node divided differences.
- **Runge Phenomenon Analysis:** Visualizes edge oscillations on $f(x) = \frac{1}{1 + 25x^2}$ and demonstrates convergence mitigation via Chebyshev nodes.
- **Automated Testing:** Full test suite verifying numerical consistency and edge cases using `pytest`.

## Installation

### Prerequisites
- Python 3.11+

### Install Dependencies
```bash
pip install -r requirements.txt
