import sys
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return pow(x, 2) - (10 * np.cos(np.pi * x))


def get_xs(a, b, n):
    step = (b-a)/(n-1)
    ret = []
    for i in range(n):
        ret.append(a)
        a += step
    return ret


def get_ys(xs):
    return [f(x) for x in xs]


def spline3(x_points, y_points, xs, boundary_cond):
    size = len(x_points) - 2
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 4
            if j == i+1 or j == i-1:
                matrix[i][j] = 1

    g = np.zeros(size); h = []
    for i in range(size):
        h.append(x_points[i+1] - x_points[i])
        g[i] = 6 / (h[i]**2) * (y_points[i] - 2*y_points[i+1] + y_points[i+2])
    h.append(x_points[-1] - x_points[-2])
    z = np.linalg.solve(matrix, g)

    # the boundary condition is set here
    z = list(z)
    if boundary_cond == 1:
        z = [0] + z + [0]
    else:
        z = [z[0]] + z + [z[-1]]

    a = []; b = []; c = []; d = []
    for i in range(size+1):
        a.append((z[i+1] - z[i]) / (6 * h[i]))
        b.append(0.5 * z[i])
        c.append((y_points[i+1] - y_points[i]) / h[i] - (z[i+1] + 2 * z[i]) / 6 * h[i])
        d.append(y_points[i])

    nr_fun = 0
    ys = []
    for i in range(len(xs)):
        while x_points[nr_fun + 1] < xs[i] < x_points[-1]:
            nr_fun += 1
        ys.append(get_val([d[nr_fun], c[nr_fun], b[nr_fun], a[nr_fun]], x_points[nr_fun], xs[i]))

    return ys


def spline2(x_points, y_points, xs, boundary_cond):
    size = len(x_points) - 1
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1
            if j == i-1:
                matrix[i][j] = 1

    g = np.zeros(size); h = []
    for i in range(size):
        h.append(x_points[i+1] - x_points[i])
        g[i] = 2 / h[i] * (y_points[i+1] - y_points[i])
    h.append(x_points[-1] - x_points[-2])
    b = np.linalg.solve(matrix, g)

    # the boundary condition is set here
    b = list(b)
    if boundary_cond == 1:
        b = [0] + b
    else:
        b = [b[-1]] + b

    a = []; c = []
    print(size)
    for i in range(size):
        a.append((b[i+1] - b[i]) / (2 * h[i]))
        c.append(y_points[i])

    nr_fun = 0
    ys = []
    for i in range(len(xs)):
        while x_points[nr_fun + 1] < xs[i] < x_points[-1]:
            nr_fun += 1
        ys.append(get_val([c[nr_fun], b[nr_fun], a[nr_fun]], x_points[nr_fun], xs[i]))

    return ys


def get_val(coeff, xi, x):
    val = 0
    for i, elem in enumerate(coeff):
        val += elem * (x - xi) ** i
    return val


def main():
    start = -np.pi
    end = np.pi
    n_points = 8
    n_draw = 1000
    spline = 2

    xs = get_xs(start, end, n_draw)
    ys = get_ys(xs)

    xp = get_xs(start, end, n_points)
    yp = get_ys(xp)

    plt.plot(xs, ys, 'k')
    plt.plot(xp, yp, 'k.', markersize=10)

    if spline == 2 or spline == 0:
        ys = spline2(xp, yp, xs, 1)
        plt.plot(xs, ys, 'm')
        ys = spline2(xp, yp, xs, 2)
        plt.plot(xs, ys, 'g')
    if spline == 3 or spline == 0:
        ys = spline3(xp, yp, xs, 1)
        plt.plot(xs, ys, 'b')
        ys = spline3(xp, yp, xs, 2)
        plt.plot(xs, ys, 'r')

    plt.show()


if __name__ == "__main__":
    sys.exit(main())
