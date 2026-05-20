from typing import Iterator, Any
from dataclasses import dataclass


@dataclass
class EigenSolverResult:
    eigenvalue: float
    eigenvector: list[float]
    iterations: int
    converged: bool
    history: list[float]


class BaseEigenSolver:
    def __init__(self, matrix: list[list[float]], tol: float = 1e-6, max_iter: int = 100):
        self.matrix = matrix
        self.tol = tol
        self.max_iter = max_iter
        self._history: list[float] = []
        self._converged = False
        self.method = "Base Eigen Solver"

    def solve(self) -> tuple[float, list[float]]:
        raise NotImplementedError("Subclasses must implement solve()")

    def solve_iter(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for i, val in enumerate(self._history):
            yield i, {"eigenvalue_estimate": val}

    def get_result(self) -> EigenSolverResult:
        return EigenSolverResult(
            eigenvalue=self._history[-1] if self._history else None,
            eigenvector=[],
            iterations=len(self._history),
            converged=self._converged,
            history=self._history.copy(),
        )