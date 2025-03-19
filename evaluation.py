
from typing import List, Tuple


def evaluate_cycle_prediction(
        prediction: List[List[str]], original: List[List[str]]) -> Tuple[float, float]:

    found = 0
    err = 0
    total = 0
    for i, j in zip(prediction, original):
        found += len(set(i).intersection(set(j)))
        err += len(set(i) - set(j))
        total += len(j)

    if total == 0:
        return 0, 0

    if found + err > 0:
        return found / total, err / (found + err)
    else:
        return 0.0, 0.0