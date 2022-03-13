from grid import *


# Known exact solution (analytical function)
def uexact(point):
    x = point[0]
    return math.sin(9 * (x + 0.2) ** 2)


# K(x) (known analytical function)
def kfun(point):
    x = point[0]
    return 1.0 / (18 * 18 * (x + 0.2))


# F(x) right side function (known analytical function)
def ffun(point):
    x = point[0]
    return (x + 1.2) * uexact(point)


# numerical solution: defined through solution vector u and basis functions phi_i
def unumer(point):
    global grid, u
    ielem = find_element(grid, point)
    return u(ielem)


grid = Grid()
grid.regular_grid(0, 1, 0, 1, 4, 4)
Nelem = grid.Nelem
Nvert = grid.Nvert
Nface = grid.Nface

# approximate analytic functions
fvec = np.array([ffun(grid.elem_center[i]) for i in range(Nelem)])
kvec = np.array([kfun(grid.elem_center[i]) for i in range(Nelem)])

# SLE Assembly and Solution
# left hand side matrix
M = np.zeros(Nelem, Nelem)
# right hand side vector
rhs = np.zeros(Nelem)
