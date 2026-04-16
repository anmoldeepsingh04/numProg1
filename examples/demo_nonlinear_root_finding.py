import numpy as np
import matplotlib.pyplot as plt
from numerics.nonLinearSolvers.secant import SecantSolver
from numerics.nonLinearSolvers.regula_falsi import RegulaFalsiSolver
from numerics.nonLinearSolvers.bisection import BisectionSolver
from numerics.utils.plotting import Plotter

def f(x):
    return x**3 - x - 2

solver1 = SecantSolver(f, 1, 2)
root1 = solver1.solve()
solver2 = RegulaFalsiSolver(f, 1, 2)
root2 = solver2.solve()
solver3 = BisectionSolver(f, 1, 2)
root3 = solver3.solve()

print("Root: ", root1)
print("Root: ", root2)
print("Root: ", root3)

# for steps in solver1.get_steps():
#     print(steps)

Plotter.plot_convergence(solver1.get_history(), solver1.method)
Plotter.plot_convergence(solver2.get_history(), solver2.method)
Plotter.plot_convergence(solver3.get_history(), solver3.method)

plt.show()

# Plotter.plot_function(f, 0, 3)