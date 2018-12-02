import sys
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


# set print precision
np.set_printoptions(floatmode='maxprec')


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
    df.to_excel(filename, sheet_name='sheet1', index=False, header=False)


def f(x):
    return pow(x, 2) - (10 * np.cos(np.pi * x))


def get_xs(a, b, n):
    step = (b-a)/(n-1)
    ret = []
    for i in range(n):
        ret.append(a)
        a += step
    return ret


def f_list(xs):
    ys = []
    for x in xs:
        ys.append(f(x))
    return ys


def get_x(x_points):
    size = len(x_points)
    x = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            x[i][j] = pow(x_points[i], j)
    return x


def get_b(x, y_points):
    a = np.linalg.inv(np.dot(np.transpose(x), x))
    b = np.dot(np.transpose(x), y_points)
    return np.dot(a, b)


def approximate(x_points, y_points):
    # B - wektor wspolczynnikow wielomianu b0 + b1x + b2x^2 + ...
    # B = (X^t * X)^-1 * X^t * y
    X = get_x(x_points)
    return get_b(X, y_points)


def get_val(f_coeff, x):
    val = 0
    for i, coeff in enumerate(f_coeff):
        val += f_coeff[i] * pow(x, i)
    return val


def show_fun(x_points, y_points, xs, B):
    y_app = []
    for i in range(len(xs)):
        y_app.append(get_val(B, xs[i]))

    # base function
    ys = f_list(xs)
    plt.plot(x_points, y_points, 'k.', markersize=8)
    plt.plot(xs, ys, 'k')

    # approximated function
    plt.plot(xs, y_app, 'c')

    plt.draw()
    plt.show()


def main():
    n_draw = 1000
    xs = get_xs(-np.pi, np.pi, n_draw)

    for j in range(5, 21):
        n_points = j
        x_points = get_xs(-np.pi, np.pi, n_points)
        y_points = f_list(x_points)

        b = approximate(x_points, y_points)
        show_fun(x_points, y_points, xs, b)


if __name__ == "__main__":
    sys.exit(main())