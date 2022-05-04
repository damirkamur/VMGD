import numpy as np


class D1Lin:
    def __init__(self, vertices, ibasis):
        self.nbasis = 2
        self.phi = [lambda xi: 1 - xi, lambda xi: xi]
        self.ibasis = ibasis
        L = vertices[1] - vertices[0]
        self.h = L
        self.mass = L * np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
        self.stiff = 1 / L * np.array([[1, -1], [-1, 1]])
        self.lumped_mass = L * np.array([1 / 2, 1 / 2])
        self.tran = np.array([[[-1 / 2, 1 / 2], [-1 / 2, 1 / 2]]])
        self.p0 = vertices[0]

    def to_xi(self, x):
        return (x - self.p0) / self.h
