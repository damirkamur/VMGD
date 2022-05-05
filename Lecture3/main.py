import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sin, sqrt
from grid import *

# ===== -d( K(x, y) * du(x, y)/dx )/dx - d( K(x, y) * du(x, y)/dy )/dy + u(x, y) = F(x, y)
S, H = 10, 0.2


def uexact(point_2d) -> float:
    """
    Известное аналитическое решение
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = point_2d[0]
    y = point_2d[1]
    return sin(S * (x + y + H) ** 2)


def kfun(point_2d) -> float:
    """
    Известная аналитическая функция
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = point_2d[0]
    y = point_2d[1]
    return 1 / (8 * S ** 2 * (x + y + H))


def ffun(point_2d) -> float:
    """
    Правая часть уравнения (известная аналитическая функция)
    :param point_2d: двумерная точка
    :return: значение функции
    """
    x = point_2d[0]
    y = point_2d[1]
    return (x + y + H + 1) * uexact(point_2d)


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
# grid = gu_build_from_gmsh_vtk('grid3.vtk')
# grid = gu_build_from_tuples(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)),
#                             ((0, 1, 4, 3), (1, 2, 5, 4)))
Nelemx = 150
Nelemy = 30
grid = gu_reggrid_tr(0, 0, 1, 1/50, 100, 2)
# grid = gu_build_from_tuples(((0, 0), (1, 0), (2, 0), (1, 1), (2, 1)), ((0, 1, 3), (1, 2, 4, 3)))
# 1. ============================== Входные данные и аппроксимация аналитических функций
Nelem = grid.Nelem
Nvert = grid.Nvert
Nface = grid.Nface
fvec, kvec = exact_approximate()

# 2. ============================== Сборка матрицы и решение
M = np.zeros((Nelem, Nelem))
rhs = np.zeros(Nelem)

for i in range(Nelem):
    # интеграл масс
    volume = grid.elem_volume[i]
    M[i][i] += volume

    # правая часть
    rhs[i] += volume * fvec[i]

# stiffness (внутри)
for iface in grid.internal_faces:
    ielem1 = grid.face_elem[iface][0]
    ielem2 = grid.face_elem[iface][1]
    irow1 = ielem1
    irow2 = ielem2

    k1 = kvec[ielem1]
    k2 = kvec[ielem2]
    h1 = grid.face_elem_distance[iface][0]
    h2 = grid.face_elem_distance[iface][1]
    kc = k1 * k2 * (h1 + h2) / (k1 * h2 + k2 * h1)
    h = grid.face_cross_distance[iface]
    m = kc / h * grid.face_area[iface] * grid.face_cosn[iface]

    M[irow1][irow1] += m
    M[irow1][irow2] -= m

    M[irow2][irow2] += m
    M[irow2][irow1] -= m

# ГУ первого рода
for iface in grid.boundary_faces:
    x1 = grid.face_center[iface][0]
    iloc = 0 if grid.face_elem[iface][0] > -1 else 1
    ielem = grid.face_elem[iface][iloc]
    M[ielem][ielem] += 1
    pnt = grid.face_center[iface]
    rhs[ielem] += uexact(pnt)

# # stiffness (Условия Дирихле)
# for iface in grid.boundary_faces:
#     x1 = grid.face_center[iface][0]
#     iloc = 0 if grid.face_elem[iface][0] > -1 else 1
#     ielem = grid.face_elem[iface][iloc]
#
#     h = grid.face_cross_distance[iface]
#     kc = kvec[ielem]
#     m = kc / h * grid.face_area[iface] * grid.face_cosn[iface]
#     M[ielem][ielem] += m
#     pnt = grid.face_center[iface]
#     rhs[ielem] += uexact(pnt) * m
print('Сборка матрицы завершена → решение')
u = np.linalg.solve(M, rhs)
print('матрица решена → графики')
# 3. ============================== Визуализация и вывод
Nvis = 1000
x = np.linspace(0, 1, Nvis)
y = 0.0

y_exact, y_numer = np.zeros(Nvis), np.zeros(Nvis)
for i in range(Nvis):
    y_exact[i] = uexact([x[i], y])
    y_numer[i] = unumer([x[i], y])

# графики y_exact(x), y_numer(x)
plt.plot(x, y_exact, x, y_numer)
plt.legend(("EXACT", "NUMER"))
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()
plt.xlabel('x')
plt.ylabel('u')
plt.legend(('Точное решение', 'Численное решение'))
name_p_file = f'pictures/Сравнение решений. Разбиение по x({Nelemx}), разбиение по y({Nelemy}, y={y}).png'
plt.show()
# plt.savefig(name_p_file, dpi=300)

# невязка (максимальная и стандартное отклонение)
Nmax = np.max(np.abs(y_exact - y_numer))
N2 = np.std(y_exact - y_numer)
# N2A = sqrt(1 / sum(grid.elem_volume) * sum((grid.elem_volume[i] * (unumer([x[i], y]) - uexact([x[i], y])) ** 2 for i in range(Nvis))))
print(Nmax)
print(N2)
# print(N2A)
# x0, x1, y0, y1 = 0, 1, 0, 1
# Nvis = 100
# X, Y = np.meshgrid(np.linspace(x0, x1, Nvis), np.linspace(y0, y1, Nvis))
# y_numer = np.zeros((Nvis, Nvis))
# y_exact = np.zeros((Nvis, Nvis))
# for i in range(Nvis):
#     for j in range(Nvis):
#         y_exact[i][j] = uexact([X[i][j], Y[i][j]])
#         y_numer[i][j] = unumer([X[i][j], Y[i][j]])
#
# fig = plt.figure()
# axes = fig.add_subplot(projection='3d')
# axes.plot_surface(X, Y, y_exact)
# axes.plot_surface(X, Y, y_numer)
# plt.show()
