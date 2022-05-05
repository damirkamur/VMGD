import matplotlib.pyplot as plt
from math import floor
from D1Sqr import *

# ===== -d( K(x) * du(x)/dx )/dx + u(x) = F(x) ====

# 1. ===================================== Input Data
A, B = 0, 1
Nelem = 5
Nvert = Nelem + 1
Nbasis = Nelem + Nvert
grid = np.linspace(A, B, Nvert)
center = np.array([(grid[i + 1] + grid[i]) / 2 for i in range(Nelem)])

elements = list()
for ielem in range(Nelem):
    vert = [grid[ielem], grid[ielem + 1]]
    bas = [ielem, ielem + 1, Nvert + ielem]
    elements.append(D1Sqr(vert, bas))


# 2. ============================= Analytical Functions
def uexact(x: float) -> float:
    return 3 * x ** 5 - x ** 4 - x ** 3 - 3 * x ** 2 + x


def kfun(x: float) -> float:
    return 1


def ffun(x: float):
    return -(60 * x ** 3 - 12 * x ** 2 - 6 * x - 6) + uexact(x)


def find_elem(x: float) -> int:
    if x < grid[0] or x > grid[-1]:
        raise Exception('выход за расчетную область')
    return min(Nelem - 1, max(0, floor((x - grid[0]) / (grid[1] - grid[0]))))


def unumer(x: float) -> float:
    ielem = find_elem(x)
    e = elements[ielem]
    xi = e.to_xi(x)
    return sum([u[e.ibasis[i]] * e.phi[i](xi) for i in range(e.nbasis)])


fvec = list()
kvec = list()
for ielem in range(Nelem):
    kvec.append(kfun(center[ielem]))

for i in range(Nvert):
    fvec.append(ffun(grid[i]))

for i in range(Nvert, Nbasis):
    fvec.append(ffun(center[i - Nvert]))

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
rhs = Mass.dot(fvec)

# ГУ
M[0][:] = 0
M[0][0] = 1
rhs[0] = uexact(grid[0])

M[Nvert - 1][:] = 0
M[Nvert - 1][Nvert - 1] = 1
rhs[Nvert - 1] = uexact(grid[Nvert - 1])

u = np.linalg.solve(M, rhs)

# 4. ============================== Visualization and output
Nvis = 1000
x = np.linspace(A, B, Nvis)
y_exact = np.zeros(Nvis)
y_numer = np.zeros(Nvis)
for i in range(Nvis):
    y_exact[i] = uexact(x[i])
    y_numer[i] = unumer(x[i])

plt.plot(x, y_exact, x, y_numer)
plt.legend(("EXACT", "NUMER"))
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()
plt.xlabel('x')
plt.ylabel('u')
plt.show()
