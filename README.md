# numProg1 - Numerical Programming 1 Tool

Educational tool for the Numerical Programming 1 course at TUM. Implements various numerical methods with a consistent object-oriented interface and live graphing capabilities.

## Installation

```bash
pip install numprog1-tool
```

Or for development:

```bash
git clone https://github.com/user/numProg1.git
cd numProg1
pip install -e ".[dev]"
```

## Quick Start

```python
from numprog1.methods.nonlinear import BisectionSolver, SecantSolver

def f(x):
    return x**3 - x - 2

# Bisection method
solver = BisectionSolver(f, 1, 2, tol=1e-6)
root = solver.solve()
print(f"Root: {root}")

# Access iteration history for live graphing
result = solver.get_result()
print(f"History: {result.history}")
```

## Available Methods

### Nonlinear Solvers
- `BisectionSolver` - Bisection method
- `SecantSolver` - Secant method
- `RegulaFalsiSolver` - Regula Falsi method

## Live Graphing

```python
from numprog1.utils.plotting import Plotter

# Static convergence plot
Plotter.plot_convergence(history, "Bisection Method")

# Animated convergence (useful for teaching)
anim = Plotter.animate_convergence(history, errors, "Bisection Method")
```

## Project Structure

```
numprog1/
├── core/           # Base classes (BaseSolver, BaseIntegrator, etc.)
├── methods/        # Concrete implementations
│   ├── nonlinear/  # Root finding methods
│   ├── linear/     # Linear system solvers
│   ├── integration/ # Numerical integration
│   └── eigen/      # Eigenvalue solvers
├── utils/          # Plotting and helpers
└── ui/             # Web interface (Streamlit)
```

## License

MIT License