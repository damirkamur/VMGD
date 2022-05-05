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
def exact_approximate() -> (np.ndarray, np.ndarray, np.ndarray):
    fvec = np.zeros(Nelem)
    kvec = np.zeros(Nelem)
    uvec = np.zeros(Nelem)
    for i in range(Nelem):
        point = grid.elem_center[i]
        fvec[i] = ffun(point)
        kvec[i] = kfun(point)
        uvec[i] = uexact(point)
    return fvec, kvec, uvec


Nr = np.array([10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
Nt = np.array([3, 10, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250])
N2rec = np.zeros(len(Nr))
N2tr = np.zeros(len(Nt))
Nelemrec = np.zeros(len(Nr))
Nelemtr = np.zeros(len(Nt))

for index in range(len(Nr)):
    print(f'{index} rec')
    # 0. ============================== Считывание сетки
    # grid = gu_build_from_gmsh_vtk(f'grid_rec{index + 1}.vtk')
    grid = gu_reggrid(0, 0, 1, 2 / Nr[index], Nr[index], 2)

    # 1. ============================== Входные данные и аппроксимация аналитических функций
    Nelem = grid.Nelem
    Nvert = grid.Nvert
    Nface = grid.Nface
    fvec, kvec, uvec = exact_approximate()

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

    # невязка (максимальная и стандартное отклонение)
    N2 = np.std(y_exact - y_numer)
    print(N2)
    Sum_Volume = math.fabs(sum(grid.elem_volume[i] for i in range(Nelem)))
    N2A = (1 / Sum_Volume * sum((uvec[i] - u[i]) ** 2 * grid.elem_volume[i] for i in range(Nelem))) ** 0.5
    print(N2A)
    N2rec[index] = N2A
    Nelemrec[index] = Nelem

for index in range(len(Nt)):
    print(f'{index} tr')
    # 0. ============================== Считывание сетки
    # grid = gu_build_from_gmsh_vtk(f'grid_tr{index + 1}.vtk')
    grid = gu_reggrid_tr(0, 0, 1, 2 / Nt[index], Nt[index], 2)

    # 1. ============================== Входные данные и аппроксимация аналитических функций
    Nelem = grid.Nelem
    Nvert = grid.Nvert
    Nface = grid.Nface
    fvec, kvec, uvec = exact_approximate()

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

    # невязка (максимальная и стандартное отклонение)
    N2 = np.std(y_exact - y_numer)
    print(N2)
    Sum_Volume = math.fabs(sum(grid.elem_volume[i] for i in range(Nelem)))
    N2A = (1 / Sum_Volume * sum(
        (uvec[i] - u[i]) ** 2 * grid.elem_volume[i] for i in range(Nelem))) ** 0.5
    print(N2A)
    N2tr[index] = N2A
    Nelemtr[index] = Nelem

plt.plot(Nelemrec, N2rec, Nelemtr, N2tr)
# plt.loglog(Nelemrec, N2rec, Nelemtr, N2tr)
plt.legend(("RECTANGLE", "TRIANGLE"))
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()
plt.xlabel('Nelem')
name_p_file = f'pictures/НевязкаN2A_not_log.png'
plt.ylabel('N2A')
# plt.show()
plt.savefig(name_p_file, dpi=300)
