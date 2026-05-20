from typing import Callable, Iterator, Optional, Any
from dataclasses import dataclass


@dataclass
class IntegratorResult:
    value: float
    iterations: int
    history: list[float]
    errors: list[float]


class BaseIntegrator:
    def __init__(self, f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self._history: list[float] = []
        self._errors: list[float] = []
        self.method = "Base Integrator"

    def integrate(self) -> float:
        raise NotImplementedError("Subclasses must implement integrate()")

    def integrate_iter(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for i, (h, e) in enumerate(zip(self._history, self._errors)):
            yield i, {"estimate": h, "error": e}

    def get_result(self) -> IntegratorResult:
        return IntegratorResult(
            value=self._history[-1] if self._history else None,
            iterations=len(self._history),
            history=self._history.copy(),
            errors=self._errors.copy(),
        )

    def get_history(self) -> list[float]:
        return self._history.copy()