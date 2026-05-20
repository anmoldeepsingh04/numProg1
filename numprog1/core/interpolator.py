from typing import Callable, Iterator, Optional, Any
from dataclasses import dataclass


@dataclass
class InterpolatorResult:
    coefficients: list[float]
    points: list[tuple[float, float]]
    method: str


class BaseInterpolator:
    def __init__(self, x_data: list[float], y_data: list[float]):
        self.x_data = x_data
        self.y_data = y_data
        self._coefficients: list[float] = []
        self.method = "Base Interpolator"

    def interpolate(self) -> list[float]:
        raise NotImplementedError("Subclasses must implement interpolate()")

    def evaluate(self, x: float) -> float:
        raise NotImplementedError("Subclasses must implement evaluate()")

    def get_result(self) -> InterpolatorResult:
        return InterpolatorResult(
            coefficients=self._coefficients.copy(),
            points=list(zip(self.x_data, self.y_data)),
            method=self.method,
        )