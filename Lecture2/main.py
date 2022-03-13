import numpy as np
from math import sin
from matplotlib import pyplot as plt


# -d( K(x) * du(x)/dx )/dx + u(x) = F(x)


# Функции
def u_exact(x):
    return sin(s * (x + h) ** 2)


def k_fun(x):
    return 1 / ((2 * s) ** 2 * (x + h))


def f_fun(x):
    return (x + h + 1) * sin(s * (x + h) ** 2)


def u_numer(x):
    global u, n_elem
    ret = 0
    for i in range(n_elem):
        ret += u[i] * phi(i, x)
    return ret


def phi(i, x):
    if abs(x - b) < 10e-14 and i == n_elem - 1:
        return 1
    return 1 if grid[i] <= x < grid[i + 1] else 0


counter = 0
with open('output.txt', 'w') as file:
    for n_elem in range(10, 1001, 50):
        counter += 1
        # Инициализация
        h, s = 0.2, 10
        a, b = 0, 1
        n_vert = n_elem + 1
        grid = np.linspace(a, b, n_vert)
        u = np.zeros(n_elem)
        center = np.array([(grid[i + 1] + grid[i]) * 0.5 for i in range(n_elem)])

        # Решение СЛАУ
        m = np.zeros((n_elem, n_elem))
        rhs = np.zeros(n_elem)

        for i_row in range(n_elem):
            # левый сосед
            if i_row > 0:
                # внутренние элементы
                coef = k_fun(grid[i_row]) / (center[i_row] - center[i_row - 1])
                m[i_row][i_row] += coef
                m[i_row][i_row - 1] -= coef
            else:
                # левая граница - условие Дирихле
                coef = k_fun(grid[0]) / (center[0] - grid[0])
                m[i_row][i_row] += coef
                rhs[i_row] += u_exact(grid[0]) * coef

            # правый сосед
            if i_row < n_elem - 1:
                # внутренние элементы
                coef = k_fun(grid[i_row + 1]) / (center[i_row + 1] - center[i_row])
                m[i_row][i_row] += coef
                m[i_row][i_row + 1] -= coef
            else:
                # правая граница - условие Дирихле
                coef = k_fun(grid[n_vert - 1]) / (grid[n_vert - 1] - center[n_elem - 1])
                m[i_row][i_row] += coef
                rhs[i_row] += u_exact(grid[n_vert - 1]) * coef

            # интеграл масс
            length = grid[i_row + 1] - grid[i_row]
            m[i_row][i_row] += length

            # правая часть
            rhs[i_row] += length * f_fun(center[i_row])

        # Решение
        u = np.linalg.solve(m, rhs)

        # визуализация и вывод результатов
        n_vis = 1000
        x = np.linspace(a, b, n_vis)

        y_exact, y_numer = np.zeros(n_vis), np.zeros(n_vis)
        for i in range(n_vis):
            y_exact[i] = u_exact(x[i])
            y_numer[i] = u_numer(x[i])

        fig1 = plt.gcf()
        plt.title(f'Сравнение результатов. Nelem = {n_elem}')
        plt.plot(x, y_exact, x, y_numer)
        plt.legend(('точное', 'численное'))
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        fig1.savefig(f'pictures/nelem {n_elem}.png', dpi=300)
        fig1.clf()

        # Расчет невязки
        n_max = max(np.fabs(y_exact - y_numer))
        n2 = np.std(y_exact - y_numer)

        # Вывод информации
        file.write(f'{n_elem}\t{n_max}\t{n2}\n')
        print('-' * 40)
        print(f'n_elem = {n_elem}')
        print(f'n_max = {n_max}\nn2 = {n2}')

x = np.zeros(counter)
y = np.zeros(counter)
y2 = np.zeros(counter)

with open('output.txt', 'r') as file:
    for i, data in enumerate(file):
        x[i], y[i], y2[i] = map(float, data.split())

fig1 = plt.gcf()
plt.title('Сеточная сходимость')
plt.plot(x, y, x, y2)
plt.legend(('Nmax', 'N2'))
plt.minorticks_on()
plt.grid(which='major', linewidth=1)
plt.grid(which='minor', linestyle=':')
fig1.savefig('pictures/Сеточная сходимость.png', dpi=300)
