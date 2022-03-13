import numpy as np
import math


class Grid:
    def regular_grid(self, x0: float, x1: float, y0: float, y1: float, nx: int, ny: int) -> None:
        self.Nvert = (nx + 1) * (ny + 1)
        self.Nelem = nx * ny
        self.Nface = nx * (ny + 1) + ny * (nx + 1)
        self.vert = np.zeros((self.Nvert, 2))
        hx = (x1 - x0) / nx
        hy = (y1 - y0) / ny
        for i in range(ny + 1):
            for j in range(nx + 1):
                k = j + i * (nx + 1)
                self.vert[k] = [x0 + hx * j, y0 + hy * i]
        self.vert = np.array(self.vert)
        self.elem_vert = [[j + i * (nx + 1), j + 1 + i * (nx + 1), j + 1 + (i + 1) * (nx + 1), j + (i + 1) * (nx + 1)]
                          for i in range(ny) for j in range(nx)]
        self.elem_nvert = [4] * nx * ny
        self.face_elem = list()
        for i in range(ny + 1):
            for j in range(nx):
                if i == 0:
                    self.face_elem.append([-1, j + i * nx])
                elif i == ny:
                    self.face_elem.append([j + (i - 1) * nx, -1])
                else:
                    self.face_elem.append([j + (i - 1) * nx, j + i * nx])
        for i in range(nx + 1):
            for j in range(ny):
                if i == 0:
                    self.face_elem.append([-1, i + j * nx])
                elif i == nx:
                    self.face_elem.append([i - 1 + j * nx, -1])
                else:
                    self.face_elem.append([i - 1 + j * nx, i + j * nx])
        if nx != 1 and ny != 1:
            self.elem_center = [
                [(x1 - x0 - hx) / (nx - 1) * j + x0 + hx / 2, (y1 - y0 - hy) / (ny - 1) * i + y0 + hy / 2]
                for i in range(ny) for j in range(nx)]
        elif nx == 1 and ny != 1:
            self.elem_center = [
                [hx / 2, (y1 - y0 - hy) / (ny - 1) * i + y0 + hy / 2]
                for i in range(ny) for j in range(nx)]
        elif nx != 1 and ny == 1:
            self.elem_center = [
                [(x1 - x0 - hx) / (nx - 1) * j + x0 + hx / 2, hy / 2]
                for i in range(ny) for j in range(nx)]
        else:
            self.elem_center = [[hx / 2, hy / 2]]
        self.elem_center = np.array(self.elem_center)


def cross_product(a: 'np.array()', b: 'np.array()'):
    return a[0] * b[1] - a[1] * b[0]


def is_within_triangle(p1: 'np.array()', p2: 'np.array()', p3: 'np.array()', p_target: 'np.array()'):
    d1 = cross_product(p1 - p_target, p2 - p_target)
    d2 = cross_product(p2 - p_target, p3 - p_target)
    d3 = cross_product(p3 - p_target, p1 - p_target)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def find_element(grid: Grid, point: list[float, float]):
    for ie in range(grid.Nelem):
        e = grid.elem_vert[ie]
        ne = grid.elem_nvert[ie]
        for i in range(1, len(e) - 1):
            if is_within_triangle(grid.vert[e[0]], grid.vert[e[i]], grid.vert[e[i + 1]], point):
                return ie
    raise Exception('Вы вышли за пределы расчетной области')


