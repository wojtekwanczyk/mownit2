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


def get_x(x_points, degree):
    size = len(x_points)
    x = np.ones((size, degree))
    for i in range(size):
        for j in range(degree):
            x[i][j] = pow(x_points[i], j)
    return x


def get_b(x, y_points):
    a = np.linalg.inv(np.dot(np.transpose(x), x))
    b = np.dot(np.transpose(x), y_points)
    return np.dot(a, b)


def approximate(x_points, y_points, degree):
    # B - wektor wspolczynnikow wielomianu b0 + b1x + b2x^2 + ...
    # B = (X^t * X)^-1 * X^t * y
    X = get_x(x_points, degree)
    return get_b(X, y_points)


def get_val(f_coeff, x):
    val = 0
    for i, coeff in enumerate(f_coeff):
        val += f_coeff[i] * pow(x, i)
    return val


def show_fun(x_points, y_points, xs, B, color, title):
    y_app = []
    for i in range(len(xs)):
        y_app.append(get_val(B, xs[i]))

    # base function
    ys = f_list(xs)
    plt.plot(x_points, y_points, color + '.', markersize=10)
    plt.plot(xs, ys, 'grey')

    # approximated function
    plt.plot(xs, y_app, color, label=title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.draw()
    return y_app


def get_norm(y1, y2, which):
    if which == "max":
        return np.linalg.norm(np.subtract(y2, y1), np.inf)
    if which == "eu":
        return np.linalg.norm(np.subtract(y2, y1))


def cheby_zeros(no, a, b):
    ret = sorted(np.array([np.cos(((2 * j - 1) / (2 * (no))) * np.pi) for j in range(1, no+1)], dtype=np.float64))
    for i in range(len(ret)):
        ret[i] = 0.5 * (a+b) + 0.5 * (b-a) * ret[i]
    return ret

def get_error(y1, y2):
    err = 0
    for i in range(len(y1)):
        err += (y1[i] - y2[i])**2
    err = err / len(y1)
    return np.sqrt(err)


def main():
    n_draw = 1000
    xs = get_xs(-np.pi, np.pi, n_draw)

    n_points = 30
    degree = 3

    for j in range(10):
        x_points = get_xs(-np.pi, np.pi, n_points)
        y_points = f_list(x_points)

        x_cheby = cheby_zeros(n_points, -np.pi, np.pi)
        y_cheby = f_list(x_cheby)

        b = approximate(x_points, y_points, degree)
        b_cheby = approximate(x_cheby, y_cheby, degree)


        ys = f_list(xs)
        y_app = show_fun(x_points, y_points, xs, b, 'm', 'Wezly rownoodlegle')
        y_app_cheby = show_fun(x_cheby, y_cheby, xs, b_cheby, 'c', 'Wezly Chebysheva')

        #plt.figure(figsize=(10,10))
        plt.show()


        #norm_max = get_norm(y_app, ys, "eu")
        #norm_max_cheby = get_norm(y_app_cheby, ys, "eu")

        error = get_error(y_app, ys)
        error_cheby = get_error(y_app_cheby, ys)


        print('Równoodlegle: Liczba wezlow: ' + str(n_points) + '\tStopien: ' + str(degree) + '\tBlad sredniokwadratowy: ' + str(error))
        print('Chebysheva:   Liczba wezlow: ' + str(n_points) + '\tStopien: ' + str(degree) + '\tBlad sredniokwadratowy: ' + str(error_cheby))
        degree += 1






if __name__ == "__main__":
    sys.exit(main())