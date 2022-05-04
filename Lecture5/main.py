import numpy as np
from femutils import *
from D1Lin import *
from math import exp, floor
from copy import deepcopy
from matplotlib import pyplot as plt

# ===== dU/dt + v * dU/dx = 0
# 1. ===================================== Input Data
ndim = 1
A, B = 0, 1
Nelem = 30
Nvert = Nelem + 1
Nbasis = Nvert
grid = np.linspace(A, B, Nvert)
center = np.array([grid[i + 1] - grid[i] for i in range(Nelem)])

elements = list()
for ielem in range(Nelem):
    conn = [ielem, ielem + 1]
    vert = [grid[conn[0]], grid[conn[1]]]
    if ielem == Nelem - 1:
        conn[1] = 0
    elements.append(D1Lin(vert, conn))

# 2. ============================= Analytical Functions
v = [1]


def uexact0(x: float) -> float:
    if not -0.5 <= x <= 0.5:
        x -= x // 1
    if x > 0.5:
        x -= 1
    return exp(-(x * 10) ** 2)


def uexact(x: float, t: float) -> float:
    return uexact0(x - v[0] * t)


def ffun(x: float) -> float:
    return 0


def find_elem(x: float) -> int:
    if x < grid[0] or x > grid[-1]:
        raise Exception('выход за расчетную область')
    return min(Nelem - 1, max(0, floor((x - grid[0]) / (grid[1] - grid[0]))))


def unumer(x: float) -> float:
    ielem = find_elem(x)
    e = elements[ielem]
    xi = e.to_xi(x)
    return sum([u[e.ibasis[i]] * e.phi[i](xi) for i in range(e.nbasis)])


# 3.2 ================================ SLE Assembly and Solution
Mass = np.zeros((Nbasis, Nbasis))
Tran = np.zeros((Nbasis, Nbasis))

for ielem in range(Nelem):
    e = elements[ielem]
    for i in range(e.nbasis):
        for j in range(e.nbasis):
            Mass[e.ibasis[i]][e.ibasis[j]] += e.mass[i][j]
            for idim in range(ndim):
                Tran[e.ibasis[i]][e.ibasis[j]] += v[idim] * e.tran[idim][i][j]

# Tran = fem_algebraic_upwinding(Tran)

tend = 1.0
tau = 0.001

# Начальное значение
u = np.zeros(Nbasis)
for ivert in range(Nvert):
    u[ivert] = uexact0(grid[ivert])

M = 1 / tau * Mass + Tran

# Периодические условия
M[Nvert - 1][Nvert - 1] = 1
M[Nvert - 1][0] = -1

t = 0

while t < tend:
    t += tau
    uold = deepcopy(u)

    rhs = 1 / tau * Mass.dot(uold)
    rhs[Nvert - 1] = 0
    print(f't = {t}')
    u = np.linalg.solve(M, rhs)

# 4. ============================== Visualization and output
Nvis = 1000
x = np.linspace(A, B, Nvis)
y_exact = np.zeros(Nvis)
y_numer = np.zeros(Nvis)
for i in range(Nvis):
    y_exact[i] = uexact(x[i], t)
    y_numer[i] = unumer(x[i])

plt.plot(x, y_exact, x, y_numer)
plt.legend(("EXACT", "NUMER"))
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()
plt.xlabel('x')
plt.ylabel('u')
plt.show()
