from typing import Callable, Iterator, Any
from numprog1.core.solver import BaseSolver


class RegulaFalsiSolver(BaseSolver):
    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        super().__init__(f, tol, max_iter)
        self.a = a
        self.b = b
        self.method = "Regula-Falsi Method"

    def solve(self) -> float:
        a, b = self.a, self.b

        for i in range(self.max_iter):
            fa, fb = self.f(a), self.f(b)

            c = (a * fb - b * fa) / (fb - fa)
            fc = self.f(c)

            self._history.append(c)
            self._errors.append(abs(fc))
            self.log_step(iter=i, a=a, b=b, c=c, fc=fc)

            if abs(fc) < self.tol:
                self._converged = True
                return c

            if fa * fb < 0:
                b = c
            else:
                a = c

        self._converged = False
        return c

    def solve_iter(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for step in self._steps:
            yield step["iter"], step