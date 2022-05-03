import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.sparse import linalg
from math import sin
from grid import *

# ===== -d( K(x, y) * du(x, y)/dx )/dx - d( K(x, y) * du(x, y)/dy )/dy + u(x, y) = F(x, y)
S_con, H_con = 10, 0.2


def uexact(point_2d) -> float:
    """
    Известное аналитическое решение
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = float(point_2d[0])
    y = float(point_2d[1])
    return sin(S_con * (x + y + H_con) ** 2)


def kfun(point_2d) -> float:
    """
    Известная аналитическая функция
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = point_2d[0]
    y = point_2d[1]
    return 1 / (8 * S_con ** 2 * (x + y + H_con))


def ffun(point_2d) -> float:
    """
    Правая часть уравнения (известная аналитическая функция)
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = point_2d[0]
    y = point_2d[1]
    return (x + y + H_con + 1) * uexact(point_2d)


def unumer(point_2d) -> float:
    """
    Численное решение
    :param point_2d: двумерная точка
    :return: значение функции
    """
    point_2d = np.array(point_2d)
    ielem = find_elem(grid, point_2d)
    return u[ielem]


# аппроксимация аналитических функций
def exact_approximate() -> (np.ndarray, np.ndarray):
    fvec = np.zeros(Nelem)
    kvec = np.zeros(Nelem)
    for i in range(Nelem):
        point = grid.elem_center[i]
        fvec[i] = ffun(point)
        kvec[i] = kfun(point)
    return fvec, kvec


# 0. ============================== Считывание сетки
filename = 'gridT2.vtk'
grid = gu_build_from_gmsh_vtk(filename)
# grid = gu_build_from_tuples(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)),
#                             ((0, 1, 4, 3), (1, 2, 5, 4)))
# grid = gu_reggrid(0, 0, 1, 0.1, 100, 10)
# grid = gu_build_from_tuples(((0, 0), (1, 0), (2, 0), (1, 1), (2, 1)), ((0, 1, 3), (1, 2, 4, 3)))
# 1. ============================== Входные данные и аппроксимация аналитических функций
Nelem = grid.Nelem
Nvert = grid.Nvert
Nface = grid.Nface
fvec, kvec = exact_approximate()
fvert = [ffun(grid.vert[i]) for i in range(Nvert)]
uvert = [uexact(grid.vert[i]) for i in range(Nvert)]

# 2. ============================== Сборка матрицы и решение
data, row_ind, col_ind = list(), list(), list()
rhs = np.zeros(Nvert)

for ielem in range(Nelem):
    J11 = grid.vert[grid.elem_vert[ielem][1]][0] - grid.vert[grid.elem_vert[ielem][0]][0]
    J12 = grid.vert[grid.elem_vert[ielem][2]][0] - grid.vert[grid.elem_vert[ielem][0]][0]
    J21 = grid.vert[grid.elem_vert[ielem][1]][1] - grid.vert[grid.elem_vert[ielem][0]][1]
    J22 = grid.vert[grid.elem_vert[ielem][2]][1] - grid.vert[grid.elem_vert[ielem][0]][1]
    J = abs(J11 * J22 - J12 * J21)
    vertexes = grid.elem_vert[ielem]
    M = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * J / 24
    S = np.array([[(J21 - J22) ** 2 + (J12 - J11) ** 2, J22 * (J21 - J22) - J12 * (J12 - J11),
                   -J21 * (J21 - J22) + J11 * (J12 - J11)],
                  [J22 * (J21 - J22) - J12 * (J12 - J11), J22 ** 2 + J12 ** 2, -J22 * J21 - J12 * J11],
                  [-J21 * (J21 - J22) + J11 * (J12 - J11), -J22 * J21 - J12 * J11, J11 ** 2 + J21 ** 2]]) * kvec[
            ielem] / 2 / J
    f = np.array([fvert[k] for k in vertexes])
    SM = S + M
    Mf = M.dot(f)

    # ГУ
    if vertexes[0] in grid.boundary_vert:
        Mf[1] -= SM[1][0] * uexact(grid.vert[vertexes[0]])
        Mf[2] -= SM[2][0] * uexact(grid.vert[vertexes[0]])
    if vertexes[1] in grid.boundary_vert:
        Mf[0] -= SM[0][1] * uexact(grid.vert[vertexes[1]])
        Mf[2] -= SM[2][1] * uexact(grid.vert[vertexes[1]])
    if vertexes[2] in grid.boundary_vert:
        Mf[0] -= SM[0][2] * uexact(grid.vert[vertexes[2]])
        Mf[1] -= SM[1][2] * uexact(grid.vert[vertexes[2]])
    if vertexes[0] in grid.boundary_vert:
        SM[0] = [1, 0, 0]
        Mf[0] = uexact(grid.vert[vertexes[0]])
        SM[1][0] = 0
        SM[2][0] = 0
    if vertexes[1] in grid.boundary_vert:
        SM[1] = [0, 1, 0]
        Mf[1] = uexact(grid.vert[vertexes[1]])
        SM[0][1] = 0
        SM[2][1] = 0
    if vertexes[2] in grid.boundary_vert:
        SM[2] = [0, 0, 1]
        Mf[2] = uexact(grid.vert[vertexes[2]])
        SM[0][2] = 0
        SM[1][2] = 0


    # 11
    row_ind.append(vertexes[0])
    col_ind.append(vertexes[0])
    data.append(SM[0][0])
    # 22
    row_ind.append(vertexes[1])
    col_ind.append(vertexes[1])
    data.append(SM[1][1])
    # 33
    row_ind.append(vertexes[2])
    col_ind.append(vertexes[2])
    data.append(SM[2][2])
    # 12 21
    row_ind.append(vertexes[0])
    col_ind.append(vertexes[1])
    row_ind.append(vertexes[1])
    col_ind.append(vertexes[0])
    data.append(SM[0][1])
    data.append(SM[1][0])
    # 13 31
    row_ind.append(vertexes[0])
    col_ind.append(vertexes[2])
    row_ind.append(vertexes[2])
    col_ind.append(vertexes[0])
    data.append(SM[0][2])
    data.append(SM[2][0])
    # 23 32
    row_ind.append(vertexes[1])
    col_ind.append(vertexes[2])
    row_ind.append(vertexes[2])
    col_ind.append(vertexes[1])
    data.append(SM[1][2])
    data.append(SM[2][1])
    # rhs
    rhs[vertexes[0]] += Mf[0]
    rhs[vertexes[1]] += Mf[1]
    rhs[vertexes[2]] += Mf[2]

sA = sparse.csc_matrix((tuple(data), (tuple(row_ind), tuple(col_ind))), shape=(Nvert, Nvert))
print('Сборка матрицы завершена → решение')
u = linalg.spsolve(sA, rhs)
print('матрица решена → графики')
# 3. ============================== Визуализация и вывод

N = sum([abs(u[i] - uvert[i]) for i in range(Nvert)]) / Nvert
print(N)

# Сохранение в vtk
with open(filename, 'r') as file:
    with open(f'export/{filename[:-4]}_export_gu.vtk', 'w') as file1:
        lines = file.read()
        file1.writelines(lines)
        file1.write(f'\n')
        file1.write(f'POINT_DATA {Nvert} \n')
        file1.write(f'SCALARS sample_scalars float 1 \n')
        file1.write(f'LOOKUP_TABLE my_table \n')
        for i in range(Nvert):
            file1.write(f'{u[i]}\n')

with open(filename, 'r') as file:
    with open(f'export/{filename[:-4]}_export_exact_gu.vtk', 'w') as file1:
        lines = file.read()
        file1.writelines(lines)
        file1.write(f'\n')
        file1.write(f'POINT_DATA {Nvert} \n')
        file1.write(f'SCALARS sample_scalars float 1 \n')
        file1.write(f'LOOKUP_TABLE my_table \n')
        for i in range(Nvert):
            file1.write(f'{uvert[i]}\n')
