import numpy as np


def read_vtk_old(file_name) -> tuple[np.ndarray, np.ndarray]:
    vert_count = 0
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('POINTS'):
                vert_count = int(line.split()[1])
                break
        vert = np.zeros((vert_count, 2))
        for i, line in enumerate(file):
            if i == vert_count:
                break
            vert[i] = list(map(float, line.split()[:2]))
        elem_vert_count = int(file.readline().split()[1])
        for i, line in enumerate(file):
            if i == 0:
                elem_vert = np.zeros((elem_vert_count, int(line.split()[0])), dtype=np.int)
            elif i == elem_vert_count:
                break
            elem_vert[i] = list(map(float, line.split()[1:]))
    return vert, elem_vert


def read_vtk(file_name) -> tuple[np.ndarray, np.ndarray]:
    vert_count = 0
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('POINTS'):
                vert_count = int(line.split()[1])
                break
        vert = np.zeros((vert_count, 2))
        for i, line in enumerate(file):
            if i == vert_count:
                break
            vert[i] = list(map(float, line.split()[:2]))
        elem_vert = list()
        file.readline()
        while True:
            line = file.readline()
            if line == '\n':
                break
            info = list(map(int, line.split()))
            if info[0] < 3:
                continue
            elem_vert.append(info[1:])
    return vert, elem_vert


if __name__ == '__main__':
    a, b = read_vtk('grid2.vtk')
    print(a, b)
