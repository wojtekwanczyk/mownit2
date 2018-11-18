import multiprocessing
import sys
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import math


# here change computation precision
type_object = np.float64()
my_type = type(type_object)


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
    df.to_excel(filename, sheet_name='sheet1', index=False, header=False)


def show_fun(f, a, b, step, xl, yl, color):
    xs = []
    ys = []
    while a < b:
        xs.append(a)
        ys.append(f(a))
        a += step

    plt.plot(xs, ys, color)
    #plt.plot(xs, ys, 'b.')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.draw()


def interpolate_newton(cs, xs, x):
    ret = cs[0]
    for i in range(1, len(cs)):
        fac = 1
        for j in range(i):
            fac *= (x - xs[j])
        ret += cs[i] * fac
    return ret


def interpolate_newton_get_vals(x, y):
    cs = []
    cs.append(y[0])
    for i in range(1, len(x)):
        den = 1
        for j in range(i):
            den *= (x[i] - x[j])
        cs.append((y[i] - interpolate_newton(cs, x, x[i])) / den)
    #print(cs)
    return cs


def czeby_zeros(k):
    ret = []
    for j in range(1, k+1):
        ret.append(math.cos((2*j - 1) / (2 * k) * math.pi))
    print(ret)


def intrepolate_lagrange(xs, ys, x):
    ls = []
    ret = 0
    for i in range(len(xs)):
        ls.append(1)
        for j in range(len(xs)):
            if j != i:
                ls[i] *= (x-xs[j]) / (xs[i] - xs[j])
        ret += ys[i] * ls[i]
    return ret


def main():
    x = [5, -7, -6, 0]
    y = [1, -23, -54, -954]
    c1 = interpolate_newton_get_vals(x, y)
    # print(czeby_zeros(10))

    beg = -7.1
    end = 5.1
    step = 0.3

    show_fun(lambda ar: interpolate_newton(c1, x, ar), beg, end, step, 'xxx', 'yyy', 'r-')
    show_fun(lambda ar: intrepolate_lagrange(x, y, ar), beg, end, step, 'xxx', 'yyy', 'g-')
    plt.show()


if __name__ == "__main__":
    sys.exit(main())