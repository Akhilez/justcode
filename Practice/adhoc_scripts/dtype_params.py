def func(a: int, b: int) -> int:
    return a + b


added = func(1, 2)

print(added)


def matmul(x, y):
    return x @ y

import numpy as np
print(matmul(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])))
