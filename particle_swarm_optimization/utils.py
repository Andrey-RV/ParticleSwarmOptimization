import numpy as np
from typing import Any, Callable


def fit_function(x: np.ndarray[int | float, Any]) -> int | float:
    return x[0] ** 2 - 4 * x[0] + x[1] ** 2 - 4 * x[1] + 4 + np.sin(x[0] * x[1])


def create_inertia_weight_function(start_weight: float, end_weight: float, alpha: float) -> Callable[[float], float]:
    def inertia_weight(t: float) -> float:
        return start_weight - (start_weight - end_weight) * np.exp(-alpha * t)
    return inertia_weight
