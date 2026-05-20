"""
Example demonstrating numerical integration methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from numprog1.methods.integration import (
    TrapezoidalRule,
    SimpsonsRule,
    SimpsonsThreeEighthsRule,
    BoolesRule,
    GaussLegendreQuadrature
)
from numprog1.utils.plotting import Plotter


def f(x):
    """Test function: f(x) = x^3 - 2x + 1"""
    return x**3 - 2*x + 1


def main():
    # Integration limits
    a, b = 0, 2
    
    # Exact integral of x^3 - 2x + 1 from 0 to 2:
    # ∫(x^3 - 2x + 1)dx = (x^4/4 - x^2 + x) from 0 to 2
    # = (16/4 - 4 + 2) - (0) = (4 - 4 + 2) = 2
    exact_integral = 2.0
    
    print(f"Integrating f(x) = x^3 - 2x + 1 from {a} to {b}")
    print(f"Exact integral: {exact_integral}")
    print("=" * 50)
    
    # Test different integration methods
    methods = [
        ("Trapezoidal Rule", TrapezoidalRule(f, a, b, n=4)),
        ("Simpson's 1/3 Rule", SimpsonsRule(f, a, b, n=4)),
        ("Simpson's 3/8 Rule", SimpsonsThreeEighthsRule(f, a, b, n=3)),
        ("Boole's Rule", BoolesRule(f, a, b, n=4)),
        ("Gauss-Legendre (n=5)", GaussLegendreQuadrature(f, a, b, n=5))
    ]
    
    results = []
    for name, method in methods:
        result = method.integrate()
        error = abs(result - exact_integral)
        results.append((name, result, error))
        print(f"{name:20}: {result:.8f} (Error: {error:.2e})")
    
    print("\n" + "=" * 50)
    
    # Show convergence for Trapezoidal rule
    print("\nTrapezoidal Rule Convergence:")
    trap = TrapezoidalRule(f, a, b)
    for i, (iter_count, estimate) in enumerate(trap.integrate_iter()):
        if i >= 5:  # Show first 5 iterations
            break
        error = abs(estimate - exact_integral)
        print(f"Iteration {iter_count:2}: {estimate:.8f} (Error: {error:.2e})")
    
    # Plot function and approximation
    x_vals = np.linspace(a, b, 100)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(12, 8))
    
    # Plot function
    plt.subplot(2, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x³ - 2x + 1')
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function and Integral Area')
    plt.grid(True)
    plt.legend()
    
    # Plot convergence
    plt.subplot(2, 2, 2)
    trap = TrapezoidalRule(f, a, b)
    history = []
    for i, (iter_count, estimate) in enumerate(trap.integrate_iter()):
        history.append(estimate)
        if i >= 8:  # Show first 9 iterations
            break
    plt.plot(range(len(history)), history, 'ro-', linewidth=2, markersize=4)
    plt.axhline(y=exact_integral, color='g', linestyle='--', label=f'Exact: {exact_integral}')
    plt.xlabel('Iteration')
    plt.ylabel('Integral Estimate')
    plt.title('Trapezoidal Rule Convergence')
    plt.grid(True)
    plt.legend()
    
    # Plot comparison of methods
    plt.subplot(2, 2, 3)
    method_names = [r[0] for r in results]
    method_values = [r[1] for r in results]
    errors = [r[2] for r in results]
    
    x_pos = np.arange(len(method_names))
    plt.bar(x_pos, method_values, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=exact_integral, color='red', linestyle='-', linewidth=2, label=f'Exact: {exact_integral}')
    plt.xlabel('Integration Method')
    plt.ylabel('Integral Value')
    plt.title('Comparison of Integration Methods')
    plt.xticks(x_pos, method_names, rotation=15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot errors (log scale)
    plt.subplot(2, 2, 4)
    plt.bar(x_pos, errors, alpha=0.7, color='lightcoral', edgecolor='darkred')
    plt.yscale('log')
    plt.xlabel('Integration Method')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error Comparison of Integration Methods')
    plt.xticks(x_pos, method_names, rotation=15)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()