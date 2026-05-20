"""
Demo script for the new numprog1 API.

This script demonstrates the usage of the refactored numerical solvers
with live graphing capabilities.
"""

import matplotlib.pyplot as plt
from numprog1.methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver
from numprog1.utils.plotting import Plotter


def f(x):
    return x**3 - x - 2


def main():
    # Solve using different methods
    solvers = [
        ("Bisection", BisectionSolver(f, 1, 2, tol=1e-6, max_iter=50)),
        ("Secant", SecantSolver(f, 1, 2, tol=1e-6, max_iter=10)),
        ("Regula Falsi", RegulaFalsiSolver(f, 1, 2, tol=1e-6, max_iter=50)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, solver) in zip(axes, solvers):
        root = solver.solve()
        result = solver.get_result()

        print(f"{name} Method:")
        print(f"  Root: {root:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Converged: {result.converged}")
        print()

        # Plot convergence
        ax.plot(result.history, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Root Estimate")
        ax.set_title(f"{name} - Convergence")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Demonstrate live animation
    print("\nGenerating live animation for Bisection method...")
    bisection = BisectionSolver(f, 1, 2, tol=1e-6, max_iter=20)
    bisection.solve()
    result = bisection.get_result()

    anim = Plotter.animate_convergence(
        result.history, result.errors, bisection.method, interval=500
    )
    plt.show()


if __name__ == "__main__":
    main()