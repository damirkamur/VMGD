import numpy as np


class D2_Lin_Tr:
    """
      2
     |    \
     |      \     <- Элемент
     |        \
     0 ------- 1
    """

    def __init__(self, vertices, ibasis):
        vertices = np.array(vertices)
        self.nbasis = 3
        self.phi = [lambda xi: 1 - xi[0] - xi[1],
                    lambda xi: xi[0],
                    lambda xi: xi[1]]
        p0 = vertices[0][:]
        p1 = vertices[1][:]
        p2 = vertices[2][:]

        x1, x2, x3 = p0[0], p1[0], p2[0]
        y1, y2, y3 = p0[1], p1[1], p2[1]

        J11 = x2 - x1
        J12 = x3 - x1
        J21 = y2 - y1
        J22 = y3 - y1
        J = abs(J11 * J22 - J12 * J21)

        self.mass = np.array([[2, 1, 1],
                              [1, 2, 1],
                              [1, 1, 2]]) * J / 24
        self.stiff = np.array([
            [(J21 - J22) ** 2 + (J12 - J11) ** 2, J22 * (J21 - J22) - J12 * (J12 - J11),
             -J21 * (J21 - J22) + J11 * (J12 - J11)],
            [J22 * (J21 - J22) - J12 * (J12 - J11), J22 ** 2 + J12 ** 2, -J22 * J21 - J12 * J11],
            [-J21 * (J21 - J22) + J11 * (J12 - J11), -J22 * J21 - J12 * J11, J11 ** 2 + J21 ** 2]]) / 2 / J
        self.ibasis = ibasis
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

    def to_xi(self, point):
        x = point[0]
        y = point[1]
        xi = -(self.x1 * y - self.x3 * y - x * self.y1 + self.x3 * self.y1 + x * self.y3 - self.x1 * self.y3) / (
                self.x2 * self.y1 - self.x3 * self.y1 - self.x1 * self.y2 + self.x3 * self.y2 + self.x1 * self.y3 - self.x2 * self.y3)
        eta = -(-self.x1 * y + self.x2 * y + x * self.y1 - self.x2 * self.y1 - x * self.y2 + self.x1 * self.y2) / (
                self.x2 * self.y1 - self.x3 * self.y1 - self.x1 * self.y2 + self.x3 * self.y2 + self.x1 * self.y3 - self.x2 * self.y3)
        return np.array([xi, eta])



