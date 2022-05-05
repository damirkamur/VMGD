import numpy as np


class D2Lin_Quad:
    """
     3---------2
     |         |   <- Элемент
     |         |
     0 ------- 1
    """

    def __init__(self, vertices, ibasis):
        vertices = np.array(vertices)
        self.nbasis = 4
        self.phi = [lambda xi: (1 - xi[0]) * (1 - xi[1]),
                    lambda xi: xi[0] * (1 - xi[1]),
                    lambda xi: xi[0] * xi[1],
                    lambda xi: (1 - xi[0]) * xi[1]]
        p0 = vertices[0][:]
        v2 = vertices[1][:] - p0
        v3 = vertices[2][:] - p0
        v4 = vertices[3][:] - p0
        x2, x3, x4 = v2[0], v3[0], v4[0]
        y2, y3, y4 = v2[1], v3[1], v4[1]

        jac = lambda xi, eta: [[x2 * (1 - eta) + x3 * eta - x4 * eta, x2 * (-xi) + x3 * xi + x4 * (1 - xi)],
                               [y2 * (1 - eta) + y3 * eta - y4 * eta, y2 * (-xi) + y3 * xi + y4 * (1 - xi)]]
        dxi = [lambda xi, eta: eta - 1,
               lambda xi, eta: 1 - eta,
               lambda xi, eta: eta,
               lambda xi, eta: -eta]
        deta = [lambda xi, eta: xi - 1,
                lambda xi, eta: -xi,
                lambda xi, eta: xi,
                lambda xi, eta: 1 - xi]
        px = np.array([0.774596669241483e0, -0.774596669241483e0, 0.774596669241483e0,
                       - 0.774596669241483e0, 0.774596669241483e0, -0.774596669241483e0,
                       0.0e0, 0.0e0, 0.0e0])
        py = np.array([0.774596669241483e0, 0.774596669241483e0, -0.774596669241483e0,
                       - 0.774596669241483e0, 0.0e0, 0.0e0,
                       0.774596669241483e0, -0.774596669241483e0, 0.0e0])

        w = np.array([0.308641975308642e0, 0.308641975308642e0, 0.308641975308642e0,
                      0.308641975308642e0, 0.493827160493827e0, 0.493827160493827e0,
                      0.493827160493827e0, 0.493827160493827e0, 0.790123456790123e0])
        self.mass = np.zeros((4, 4))
        self.stiff = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                mass_val = np.zeros(9)
                stiff_val = np.zeros(9)
                for k in range(9):
                    xi = (px[k] + 1) / 2
                    eta = (py[k] + 1) / 2

                    J = jac(xi, eta)
                    modJ = np.linalg.det(J)
                    d_i_dxi = dxi[i](xi, eta)
                    d_j_dxi = dxi[j](xi, eta)
                    d_i_deta = deta[i](xi, eta)
                    d_j_deta = deta[j](xi, eta)

                    mass_val[k] = self.phi[i]([xi, eta]) * self.phi[j]([xi, eta]) * modJ

                    dx_i = J[1][1] * d_i_dxi - J[1][0] * d_i_deta
                    dx_j = J[1][1] * d_j_dxi - J[1][0] * d_j_deta
                    dy_i = -J[0][1] * d_i_dxi + J[0][0] * d_i_deta
                    dy_j = -J[0][1] * d_j_dxi + J[0][0] * d_j_deta
                    stiff_val[k] = (dx_i * dx_j + dy_i * dy_j) / modJ
                self.mass[i][j] = self.mass[j][i] = mass_val.dot(w) / 4
                self.stiff[i][j] = self.stiff[j][i] = stiff_val.dot(w) / 4
        self.ibasis = ibasis
        self.p0 = p0
        self.inv_jac = lambda xi: np.linalg.inv(jac(xi[0], xi[1]))

    def to_xi(self, x):
        x -= self.p0
        return self.inv_jac(x).dot(x)
