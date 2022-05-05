from D2Lin_Quad import *
from D2Lin_Tr import *
from grid import *
from math import sin

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
    ielem = find_elem(grid, point_2d)
    e = elements[ielem]
    xi = e.to_xi(point_2d)
    return sum([u[e.ibasis[i]] * e.phi[i](xi) for i in range(e.nbasis)])


# 0. ============================== Считывание сетки
filename = 'gridTR1.vtk'
grid = gu_build_from_gmsh_vtk(filename)
# grid = gu_reggrid_tr(0, 0, 1, 1, 2, 2)
Nelem = grid.Nelem
Nvert = grid.Nvert
Nbasis = Nvert

kvec = [kfun(grid.elem_center[i]) for i in range(Nelem)]
fvec = [ffun(grid.elem_center[i]) for i in range(Nelem)]

fvert = [ffun(grid.vert[i]) for i in range(Nvert)]
uvert = [uexact(grid.vert[i]) for i in range(Nvert)]

elements = list()
for ielem in range(Nelem):
    nvert = grid.elem_nvert[ielem]
    vertexes = grid.elem_vert[ielem]
    if nvert == 3:
        x1, y1 = grid.vert[vertexes[0]]
        x2, y2 = grid.vert[vertexes[1]]
        x3, y3 = grid.vert[vertexes[2]]
        elements.append(D2_Lin_Tr([[x1, y1], [x2, y2], [x3, y3]], vertexes))
    elif nvert == 4:
        x1, y1 = grid.vert[vertexes[0]]
        x2, y2 = grid.vert[vertexes[1]]
        x3, y3 = grid.vert[vertexes[2]]
        x4, y4 = grid.vert[vertexes[3]]
        elements.append(D2Lin_Quad([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], vertexes))
    else:
        raise Exception('Сетка содержит не только треугольники и четырехугольники')

# 3.2 ================================ SLE Assembly and Solution
# (M + S) u = Mf

Mass = np.zeros((Nbasis, Nbasis))
Stiff = np.zeros((Nbasis, Nbasis))

for ielem in range(Nelem):
    e = elements[ielem]
    for i in range(e.nbasis):
        for j in range(e.nbasis):
            Mass[e.ibasis[i]][e.ibasis[j]] += e.mass[i][j]
            Stiff[e.ibasis[i]][e.ibasis[j]] += kvec[ielem] * e.stiff[i][j]

M = Mass + Stiff
rhs = Mass.dot(fvert)

# ГУ
for ibound in grid.boundary_vert:
    M[ibound][:] = 0
    M[ibound][ibound] = 1
    rhs[ibound] = uvert[ibound]

u = np.linalg.solve(M, rhs)

print('Система решена → выгрузка')
# Сохранение в vtk
with open(filename, 'r') as file:
    with open(f'export/{filename[:-4]}_export.vtk', 'w') as file1:
        lines = file.read()
        file1.writelines(lines)
        file1.write(f'\n')
        file1.write(f'POINT_DATA {Nvert} \n')
        file1.write(f'SCALARS sample_scalars float 1 \n')
        file1.write(f'LOOKUP_TABLE my_table \n')
        for i in range(Nvert):
            file1.write(f'{u[i]}\n')

with open(filename, 'r') as file:
    with open(f'export/{filename[:-4]}_export_exact.vtk', 'w') as file1:
        lines = file.read()
        file1.writelines(lines)
        file1.write(f'\n')
        file1.write(f'POINT_DATA {Nvert} \n')
        file1.write(f'SCALARS sample_scalars float 1 \n')
        file1.write(f'LOOKUP_TABLE my_table \n')
        for i in range(Nvert):
            file1.write(f'{uvert[i]}\n')

print('программа завершила работу')

print(np.max(np.abs(np.array(uvert) - u)))
