from typing import Callable, Iterator, Optional, Any
from dataclasses import dataclass


@dataclass
class SolverResult:
    root: float
    iterations: int
    converged: bool
    history: list[float]
    errors: list[float]
    steps: list[dict[str, Any]]


class BaseSolver:
    def __init__(self, f: Callable[[float], float], tol: float = 1e-6, max_iter: int = 100):
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self._history: list[float] = []
        self._errors: list[float] = []
        self._steps: list[dict[str, Any]] = []
        self._converged = False
        self.method = "Base Solver"

    def log_step(self, **kwargs) -> None:
        self._steps.append(kwargs)

    def solve(self) -> float:
        raise NotImplementedError("Subclasses must implement solve()")

    def solve_iter(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for i, step in enumerate(self._steps):
            yield i, step

    def get_result(self) -> SolverResult:
        return SolverResult(
            root=self._history[-1] if self._history else None,
            iterations=len(self._history),
            converged=self._converged,
            history=self._history.copy(),
            errors=self._errors.copy(),
            steps=self._steps.copy(),
        )

    def get_history(self) -> list[float]:
        return self._history.copy()

    def get_steps(self) -> list[dict[str, Any]]:
        return self._steps.copy()

    def get_errors(self) -> list[float]:
        return self._errors.copy()