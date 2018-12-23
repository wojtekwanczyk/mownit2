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
    #return np.exp(1 * np.cos(6 * x))


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


def calc_ak(xs, ys, k):
    res = 0
    n = len(xs)
    for i in range(n):
        res += ys[i] * np.cos(k * xs[i])
    return (2/(n+1)) * res


def calc_bk(xs, ys, k):
    res = 0
    n = len(xs)
    for i in range(n):
        res += ys[i] * np.sin(k * xs[i])
    return (2/(n+1)) * res


def approximate3(xp, yp, xs, m):
    ys = []
    for x in xs:
        res = calc_ak(xp, yp, 0)/2
        for k in range(1, m+1):
            res += calc_ak(xp, yp, k) * np.cos(k * x)
            res += calc_bk(xp, yp, k) * np.cos(k * x)
        ys.append(res)
    return ys


def show_app3(xp, yp, xs, m, color, title):
    y_app = approximate3(xp, yp, xs, m)

    # base function
    ys = f_list(xs)
    plt.plot(xp, yp, color + '.', markersize=10)
    plt.plot(xs, ys, 'grey', label='Funkcja aproksymowana')

    # approximated function
    plt.plot(xs, y_app, color, label='Funkcja aproksymująca')
    plt.suptitle(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.draw()
    return y_app


def main():

    n_draw = 300
    zak = np.pi
    xs = get_xs(-zak, zak, n_draw)

    n_points = 21
    degree = 20
    res = [['Liczba punktów', 'Liczba współczynników', 'Błąd średniokwadratowy']]

    for j in range(10):
        plt.figure(figsize=(13, 8))
        x_points = get_xs(-zak, zak, n_points)
        y_points = f_list(x_points)

        b = approximate(x_points, y_points, degree)
        y_app = show_fun(x_points, y_points, xs, b, 'm', 'Aproksymacja funkcji F na podstawie '
                          + str(n_points) + ' węzłów funkcją z ' + str(degree) + ' współczynnikami')

        #y_app = show_app3(x_points, y_points, xs, degree, 'm', 'Aproksymacja funkcji F na podstawie '
        #                  + str(n_points) + ' węzłów funkcją z ' + str(2 * degree) + ' współczynnikami')

        ys = f_list(xs)
        error = get_error(y_app, ys)

        print('Równoodlegle: Liczba wezlow: ' + str(n_points) + '\tStopien: ' + str(degree) + '\tBlad sredniokwadratowy: ' + str(error))

        plt.grid()
        plt.show()
        res.append([n_points, degree, error])
        n_points += 5
    save('results11', res)

# n_points > degree

if __name__ == "__main__":
    sys.exit(main())
