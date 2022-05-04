import numpy as np


class D1Sqr:
    def __init__(self, vertices: np.ndarray | list[float], ibasis: list[int]) -> None:
        self.nbasis = 3
        self.phi = [lambda xi: 2 * xi ** 2 - 3 * xi + 1,
                    lambda xi: 2 * xi ** 2 - xi,
                    lambda xi: -4 * xi ** 2 + 4 * xi]
        L = vertices[1] - vertices[0]
        self.ibasis = ibasis
        self.p0 = vertices[0]
        self.h = L
        self.mass = L / 30 * np.array([[4, -1, 2],
                                       [-1, 4, 2],
                                       [2, 2, 16]])
        self.stiff = 1 / L / 3 * np.array([[7, 1, -8],
                                           [1, 7, -8],
                                           [-8, -8, 16]])
        self.lumped_mass = L * np.array([1 / 6, 1 / 6, 2 / 3])
        self.tran = np.array([[-1 / 2, -1 / 6, 2 / 3],
                              [1 / 6, 1 / 2, -2 / 3],
                              [-2 / 3, 2 / 3, 0]])

    def to_xi(self, x):
        return (x - self.p0) / self.h
