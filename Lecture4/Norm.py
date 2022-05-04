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


kol = 3
# Nt = np.zeros(kol)
# Nt_elem = np.zeros(kol)
Nt = np.array([3, 10, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500])
Nt_elem = np.zeros(len(Nt))
NtN2A = np.zeros(len(Nt))
for index in range(len(Nt)):
    # 0. ============================== Считывание сетки
    filename = f'gridT{index + 1}__.vtk'
    # grid = gu_build_from_gmsh_vtk(filename)
    # grid = gu_build_from_tuples(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)),
    #                             ((0, 1, 4, 3), (1, 2, 5, 4)))
    # grid = gu_reggrid(0, 0, 1, 0.1, 100, 10)
    grid = gu_reggrid_tr(0, 0, 1, 2 / Nt[index], Nt[index], 2)
    # grid = gu_build_from_tuples(((0, 0), (1, 0), (2, 0), (1, 1), (2, 1)), ((0, 1, 3), (1, 2, 4, 3)))
    # 1. ============================== Входные данные и аппроксимация аналитических функций
    Nelem = grid.Nelem
    Nvert = grid.Nvert
    Nface = grid.Nface
    ML = np.zeros(Nvert)
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

    # N = sum([abs(u[i] - uvert[i]) for i in range(Nvert)]) / Nvert
    # print(N)
    Sum_Volume = math.fabs(sum(grid.elem_volume[i] for i in range(Nelem)))
    # ML = M = np.array([1, 1, 1]) * J / 6

    for ii in range(Nelem):
        e = grid.elem_vert[ii]
        for i in range(len(e)):
            ML[e[i]] += 1 / 6 * grid.elem_volume[ii]

    U = np.zeros(Nvert)
    MLU = np.zeros(Nvert)
    for i in range(Nvert):
        U[i] = uvert[i] - u[i]
        MLU[i] = ML[i] * U[i]

    N2 = (1 / Sum_Volume * sum(MLU[i] ** 2 for i in range(Nvert))) ** 0.5
    print(N2)
    NtN2A[index] = N2
    Nt_elem[index] = Nelem

# plt.plot(Nt_elem, NtN2A)
plt.loglog(Nt_elem, NtN2A)
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()
plt.xlabel('Nelem')
name_p_file = f'pictures/НевязкаN2A_log.png'
plt.ylabel('N2')
# plt.show()
plt.savefig(name_p_file, dpi=300)
