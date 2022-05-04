import numpy as np


def fem_algebraic_upwinding(Tran):
    n = Tran.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if Tran[i, j] > 0:
                diff = Tran[i, j]
                Tran[i, i] += diff
                Tran[j, j] += diff
                Tran[i, j] -= diff
                Tran[j, i] -= diff
    return Tran



