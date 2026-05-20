import pytest
import numpy as np
from numprog1.methods.nonlinear import BisectionSolver, SecantSolver, RegulaFalsiSolver


def f(x):
    return x**3 - x - 2


def test_bisection_solver():
    solver = BisectionSolver(f, 1, 2, tol=1e-6, max_iter=100)
    root = solver.solve()
    assert abs(root - 1.52138) < 1e-4
    assert solver._converged is True
    result = solver.get_result()
    assert result.converged is True
    assert len(result.history) > 0


def test_bisection_solver_invalid_interval():
    with pytest.raises(ValueError):
        solver = BisectionSolver(f, 1, 1.5, tol=1e-6, max_iter=100)
        solver.solve()


def test_secant_solver():
    solver = SecantSolver(f, 1, 2, tol=1e-6, max_iter=10)
    root = solver.solve()
    assert abs(root - 1.52138) < 1e-4
    result = solver.get_result()
    assert len(result.history) > 0


def test_regula_falsi_solver():
    solver = RegulaFalsiSolver(f, 1, 2, tol=1e-6, max_iter=100)
    root = solver.solve()
    assert abs(root - 1.52138) < 1e-4
    result = solver.get_result()
    assert len(result.history) > 0


def test_solver_history_tracking():
    solver = BisectionSolver(f, 1, 2, tol=1e-6, max_iter=50)
    solver.solve()
    history = solver.get_history()
    errors = solver.get_errors()
    steps = solver.get_steps()
    assert len(history) == len(errors)
    assert len(history) == len(steps)


def test_solver_result_dataclass():
    solver = BisectionSolver(f, 1, 2, tol=1e-6, max_iter=50)
    solver.solve()
    result = solver.get_result()
    assert result.iterations == len(result.history)
    assert result.root is not None