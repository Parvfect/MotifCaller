
from typing import List, Tuple


def evaluate_cycle_prediction(
        prediction: List[List[str]], original: List[List[str]]) -> Tuple[float, float]:

    found = 0
    err = 0
    total = 0
    for i, j in zip(prediction, original):
        total += len(j)
        for k in range(len(i)):
            if i[k] in j:
                found += 1
            else:
                err += 1

    if total == 0:
        return 0, 0

    return found / total, err / total