from typing import List, Any

import numpy
import numpy as np
from numpy import ndarray

from vtkReader import read_vtk
import math


class Grid:
    def __init__(self, vert: np.ndarray, elem_vert: np.ndarray):
        self.Nelem = len(elem_vert)
        self.Nvert = len(vert)
        self.vert = vert
        self.elem_vert = elem_vert
        self.elem_nvert = [len(mas) for mas in elem_vert]
        self.elem_volume, self.elem_center = gu_volume_center(self)
        self.vert_elem = gu_vert_elem(self)
        self.vert_vert = gu_vert_vert(self)
        self.face_vert = gu_face_vert(self)
        self.Nface = len(self.face_vert)
        self.vert_face = gu_vert_face(self)
        self.elem_face, self.face_elem = gu_elem_face(self)
        self.face_center = gu_face_center(self)
        self.face_area = gu_face_area(self)
        self.boundary_faces, self.internal_faces = gu_face_types(self)
        self.face_elem_distance = gu_face_elem_dist(self)
        self.face_cosn, self.face_cross_distance = gu_face_cosn_cross_distance(self)
        print(f"Сгенерирована сетка: {self.Nelem} elements, {self.Nvert} vertices")


def gu_build_from_gmsh_vtk(filename: str) -> Grid:
    vert, elem_vert = read_vtk(filename)
    return Grid(vert, elem_vert)


def gu_build_from_tuples(vert, elem_vert) -> Grid:
    vert = np.array(vert)
    elem_vert = np.array(elem_vert, dtype=np.int)
    return Grid(vert, elem_vert)


def gu_reggrid(x0, y0, x1, y1, nx, ny):
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    vert = np.array([[x[j], y[i]] for i in range(ny) for j in range(nx)])
    elem_vert = np.array(
        [[j + i * nx, (j + 1) + i * nx, (j + 1) + (i + 1) * nx, j + (i + 1) * nx] for i in range(ny - 1) for j in
         range(nx - 1)], dtype=np.int)
    return Grid(vert, elem_vert)


def gu_vert_elem(grid: Grid) -> list[list[int]]:
    vert_elem = dict()
    for i in range(grid.Nelem):
        for j in grid.elem_vert[i]:
            if j in vert_elem:
                vert_elem[j].append(i)
            else:
                vert_elem[j] = [i]
    v_e = []
    for i in sorted((vert_elem.keys())):
        v_e.append(vert_elem[i])
    return v_e


def gu_vert_vert(grid: Grid) -> list[list[int]]:
    vert_vert = [list() for _ in range(grid.Nvert)]
    for i in range(grid.Nelem):
        for iloc1 in range(grid.elem_nvert[i]):
            iloc2 = 0 if iloc1 == grid.elem_nvert[i] - 1 else iloc1 + 1
            v1 = grid.elem_vert[i][iloc1]
            v2 = grid.elem_vert[i][iloc2]
            vert_vert[v1].append(v2)
            vert_vert[v2].append(v1)
    for iv in range(grid.Nvert):
        vert_vert[iv] = list(set(vert_vert[iv]))
    return vert_vert


def gu_face_vert(grid: Grid) -> list[list[int]]:
    face_vert = list()
    for iv1 in range(grid.Nvert):
        for iv2 in grid.vert_vert[iv1]:
            if iv1 < iv2:
                face_vert.append([iv1, iv2])
    return face_vert


def gu_vert_face(grid: Grid) -> list[list[Any]]:
    vert_face = [list() for _ in range(grid.Nvert)]
    for iface in range(grid.Nface):
        v1 = grid.face_vert[iface][0]
        v2 = grid.face_vert[iface][1]
        vert_face[v1].append(iface)
        vert_face[v2].append(iface)
    return vert_face


def gu_elem_face(grid: Grid) -> tuple[list, list]:
    elem_face = [list() for _ in range(grid.Nelem)]
    face_elem = [[-1, -1] for _ in range(grid.Nface)]

    for i in range(grid.Nelem):
        vv = grid.elem_vert[i]
        for iloc1 in range(len(vv)):
            v1 = vv[iloc1]
            iloc2 = 0 if iloc1 == len(vv) - 1 else iloc1 + 1
            v2 = vv[iloc2]
            if v1 > v2:
                v1, v2, pos = v2, v1, False
            else:
                pos = True
            for iface in range(len(grid.vert_face[v1])):
                fc = grid.vert_face[v1][iface]
                if grid.face_vert[fc][1] == v2:
                    elem_face[i].append(fc)
                    if pos:
                        face_elem[fc][0] = i
                    else:
                        face_elem[fc][1] = i
                    break
    return elem_face, face_elem


def gu_face_center(grid: Grid) -> list[float]:
    face_center = [0 for _ in range(grid.Nface)]
    for iface in range(grid.Nface):
        v1 = grid.face_vert[iface][0]
        v2 = grid.face_vert[iface][1]
        face_center[iface] = (grid.vert[v1] + grid.vert[v2]) / 2
    return face_center


def gu_face_area(grid: Grid) -> np.ndarray:
    face_length = np.zeros(grid.Nface)
    for iface in range(grid.Nface):
        v1 = grid.face_vert[iface][0]
        v2 = grid.face_vert[iface][1]
        face_length[iface] = np.linalg.norm(grid.vert[v1] - grid.vert[v2])
    return face_length


def gu_face_types(grid: Grid) -> tuple[list, list]:
    boundary = list()
    internal = list()
    for iface in range(grid.Nface):
        e = grid.face_elem[iface]
        if -1 in e:
            boundary.append(iface)
        else:
            internal.append(iface)
    return boundary, internal


def gu_face_elem_dist(grid) -> np.ndarray:
    dist = np.zeros((grid.Nface, 2))
    for iface in range(grid.Nface):
        el = grid.face_elem[iface]
        d = [-1, -1]
        if el[0] > -1:
            d[0] = np.linalg.norm(grid.face_center[iface] - grid.elem_center[el[0]])
        if el[1] > -1:
            d[1] = np.linalg.norm(grid.face_center[iface] - grid.elem_center[el[1]])
        dist[iface] = d
    return dist


def gu_face_cosn_cross_distance(grid: Grid):
    cosn = np.zeros(grid.Nface)
    cross_distance = np.zeros(grid.Nface)
    for iface in grid.internal_faces:
        el = grid.face_elem[iface]
        p1 = grid.elem_center[el[0]]
        p2 = grid.elem_center[el[1]]
        cross_distance[iface] = np.linalg.norm(p1 - p2)

        ivert1 = grid.face_vert[iface][0]
        ivert2 = grid.face_vert[iface][1]

        vec1 = p2 - p1
        vec2 = grid.vert[ivert2] - grid.vert[ivert1]
        vec2 = gu_vec_rotate(vec2, -math.pi / 2)
        cosn[iface] = gu_vec_cos(vec1, vec2)

    for iface in grid.boundary_faces:
        iloc = 0 if grid.face_elem[iface][0] > -1 else 1

        el = grid.face_elem[iface][iloc]

        p1 = grid.elem_center[el]
        p2 = grid.face_center[iface]
        cross_distance[iface] = np.linalg.norm(p1 - p2)

        vec1 = p2 - p1
        if iloc == 1:
            vec1 *= -1

        ivert1 = grid.face_vert[iface][0]
        ivert2 = grid.face_vert[iface][1]
        vec2 = grid.vert[ivert2] - grid.vert[ivert1]
        vec2 = gu_vec_rotate(vec2, -math.pi / 2)

        cosn[iface] = gu_vec_cos(vec1, vec2)
    return cosn, cross_distance


def gu_vec_rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def gu_vec_cos(v1: np.ndarray, v2: np.ndarray) -> float:
    return v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)


def gu_volume_center(grid: Grid) -> (np.ndarray, np.ndarray):
    volumes = np.zeros(grid.Nelem)
    centers = np.zeros((grid.Nelem, 2))
    for i in range(grid.Nelem):
        vertexes = grid.elem_vert[i]
        n_vertexes = len(vertexes)
        center = np.zeros(2)
        area = 0
        for j in range(1, n_vertexes - 1):
            tri_area = gu_tri_area(grid.vert[vertexes[0]], grid.vert[vertexes[j]], grid.vert[vertexes[j + 1]])
            area += tri_area
            center += tri_area / 3 * (grid.vert[vertexes[0]] + grid.vert[vertexes[j]] + grid.vert[vertexes[j + 1]])
        volumes[i] = area
        centers[i] = center / area
    return volumes, centers


def gu_tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * cross_product(b - a, c - a)


def cross_product(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def is_within_triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p_target: np.ndarray) -> bool:
    d1 = cross_product(p1 - p_target, p2 - p_target)
    d2 = cross_product(p2 - p_target, p3 - p_target)
    d3 = cross_product(p3 - p_target, p1 - p_target)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def find_elem(grid: Grid, point: np.ndarray) -> int:
    for i in range(grid.Nelem):
        vertexes = grid.elem_vert[i]
        n_vertexes = len(vertexes)
        for j in range(1, n_vertexes - 1):
            if is_within_triangle(grid.vert[vertexes[0]], grid.vert[vertexes[j]], grid.vert[vertexes[j + 1]], point):
                return i
    raise Exception(f'Вы вышли за пределы расчетной области: точка {point}')
