import sys
import matplotlib.pyplot as plt
import numpy as np


def spline3(xp, yp, xs, cond):
    n_points = len(xp) - 1
    n_plot = len(xs)

    h = [(xp[i+1] - xp[i]) for i in range(n_points)]
    z = solve(h, yp, n_points + 1)

    '''
    if cond == 1:
        z[0] = 0
        z[n_points] = 0
    else:
        z[0] = z[1]
        z[n_points] = z[n_points-1]
    '''

    def S(i, x):
        p1 = z[i+1] * (x - xp[i])**3 / (6 * h[i]) + z[i] * (xp[i+1] - x)**3 / (6 * h[i])
        p2 = ((yp[i+1] / h[i]) - (h[i] * z[i+1] / 6)) * (x - xp[i])
        p3 = ((yp[i] / h[i]) - h[i] * z[i] / 6) * (xp[i+1] - x)
        return p1 + p2 + p3

    ys = np.zeros(n_plot)

    for i, xi in enumerate(xs):
        for j, ti in enumerate(xp):
            if xi < ti:
                ys[i] = S(j - 1, xi)
                break
    ys[0] = yp[0]

    return ys


def solve(h, y, n):
    u = np.zeros(n)
    v = np.zeros(n)

    for i in range(1, n-1):
        u[i] = 2 * (h[i-1] + h[i])
        v[i] = 6 * ((y[i+1] - y[i]) / h[i]) - ((y[i] - y[i-1]) / h[i-1])

    for i in range(2, n-1):
        u[i] -= h[i-1]**2 / u[i-1]
        v[i] -= h[i-1] * v[i-1] / u[i-1]

    z = np.zeros(n)
    z[n-2] = v[n-2] / u[n-2]
    z[0] = 0
    z[n-1] = 10
    for i in range(n-2, 0, -1):
        z[i] = (v[i] - h[i] * z[i+1]) / u[i]

    return z


def spline2(xp, yp, xs, z0):
    n = len(xp) - 1
    nx = len(xs)
    ys = np.zeros(nx)


    z = np.zeros(n+1)
    z[0] = z0
    for i in range(1, n+1):
        z[i] = -z[i-1] + 2 * ((yp[i] - yp[i-1]) / (xp[i] - xp[i-1]))

    point = (lambda x, c:
         ((z[c+1] - z[c]) / (2 * (xp[c+1] - xp[c]))) * (x - xp[c])**2 + z[c] * (x - xp[c]) + yp[c])


    for i, xi in enumerate(xs):
        for j, pi in enumerate(xp):
            if xi < pi:
                ys[i] = point(xi, j-1)
                break

    return ys


def f(x):
    return np.sin(x) #* x**2 #x**12 - x**4 + 4


def get_xs(a, b, n):
    step = (b-a)/(n-1)
    ret = []
    for i in range(n):
        ret.append(a)
        a += step
    return ret


def get_ys(xs):
    return [f(x) for x in xs]


def main():
    start = 5
    end = 15
    n_points = 9
    n_plot = 100

    xs = get_xs(start, end, n_plot)
    ys = get_ys(xs)

    xp = get_xs(start, end, n_points)
    yp = get_ys(xp)

    plt.plot(xs, ys, 'k')
    plt.plot(xp, yp, 'k.', markersize=10)

    #ys = spline2(xp, yp, xs, 100)
    ys = spline3(xp, yp, xs, 1)

    plt.plot(xs, ys)
    plt.show()






if __name__ == "__main__":
    sys.exit(main())