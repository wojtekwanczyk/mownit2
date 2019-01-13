import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import sys

# here change computation precision
type_object = np.float64()
my_type = type(type_object)


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
    df.to_excel(filename, sheet_name='sheet1', index=False, header=False)


def show_fun(f, a, b, n, color, markers=False):
    xs = []
    ys = []
    step = (b-a) / n
    for i in range(n+1):
        xs.append(a)
        ys.append(f(a))
        a += step

    plt.grid()
    if markers:
        plt.plot(xs, ys, color, markersize=12)
        plt.xlabel('x')
        plt.ylabel('y')
    else:
        plt.plot(xs, ys, color)

    plt.draw()
    return ys


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


def interpolate_newton(cs, xs, x):
    #cs = interpolate_newton_get_vals(xs, ys)
    ret = cs[0]
    for i in range(1, len(cs)):
        fac = 1
        for j in range(i):
            fac *= (x - xs[j])
        ret += cs[i] * fac
    return ret


def cheby_zeros(no, a, b):
    ret = sorted(np.array([np.cos(((2 * j - 1) / (2 * (no))) * np.pi) for j in range(1, no+1)], dtype=np.float64))
    for i in range(len(ret)):
        ret[i] = 0.5 * (a+b) + 0.5 * (b-a) * ret[i]
    return ret


def points(no, a, b):
    step = (b-a)/(no-1)
    ret = []
    for i in range(no):
        ret.append(a)
        a += step
    return ret


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


def hermite2(f, df, xs, x_draw):
    y_draw = []
    for k in range(len(x_draw)):
        n = 2*len(xs)
        a = np.zeros((n, n))
        tab = []
        y = 0
        m = 1

        for i in range(len(xs)):
            tab.append(xs[i])
            tab.append(xs[i])

        for i in range(n):
            for j in range(i+1):
                if j == 0:
                    a[i][j] = f(tab[i])
                elif j == 1 & i % 2 == 1:
                    a[i][j] = df(tab[i])
                else:
                    a[i][j] = a[i][j-1] - a[i-1][j-1]
                    a[i][j] = a[i][j] / (tab[i] - tab[i-j])

        for i in range(n):
            y = y + a[i][i] * m
            m = m * (x_draw[k] - tab[i])
        y_draw.append(y)

    plt.plot(x_draw, y_draw, 'k-')
    plt.show()

    return y_draw


def f(x):
    return pow(x,2) - (10 * np.cos(np.pi * x))


def df(x):
    return 2*x + (10 * np.pi * np.sin(np.pi * x))


def f_list(xs):
    ys = []
    for x in xs:
        ys.append(f(x))
    return ys


def f_show_points(xs, k):
    ys = []
    for x in xs:
        ys.append(f(x))

    plt.plot(xs, ys, k, markersize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.draw()

def main():

    zak = np.pi

    n_draw = 500
    beg = -zak
    end = zak

    res = [['Liczba węzłów', 'Norma maksimum', 'Norma Euklidesowa']]

    for i in range(3,21):
        y1 = show_fun(f, beg, end, n_draw-1, 'b-')

        n_inter = i
        x_cheby = cheby_zeros(n_inter, -zak, zak)
        y_cheby = f_list(x_cheby)
        x_eq = points(n_inter, -zak, zak)
        y_eq = f_list(x_eq)


        # Newton

        #f_show_points(x_cheby, 'r.')
        #c1 = interpolate_newton_get_vals(x_cheby, y_cheby)
        #y2 = show_fun(lambda ar: interpolate_newton(c1, x_cheby, ar), beg, end, n_draw, 'r-')

        #print(str(n_inter) + " eu Newton(ch): " + str(np.linalg.norm(np.subtract(y2, y1))))
        #print(str(n_inter) + " max Newton(ch): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))

        #f_show_points(x_eq, 'g.')
        #c2 = interpolate_newton_get_vals(x_eq, y_eq)
        #y2 = show_fun(lambda ar: interpolate_newton(c2, x_eq, ar), beg, end, n_draw, 'g-')

        #print(str(n_inter) + " eu Newton(eq): " + str(np.linalg.norm(np.subtract(y2, y1))))
        #print(str(n_inter) + " max Newton(eq): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))



        # LagranAge

        #f_show_points(x_cheby, 'r.')
        #y2 = show_fun(lambda ar: intrepolate_lagrange(x_cheby, y_cheby, ar), beg, end, n_draw, 'r-')

        #print(str(n_inter) + " eu Lagrange(ch): " + str(np.linalg.norm(np.subtract(y2, y1))))
        #print(str(n_inter) + " max Lagrange(ch): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))

        #f_show_points(x_eq, 'y.')
        #y2 = show_fun(lambda ar: intrepolate_lagrange(x_eq, y_eq, ar), beg, end, n_draw, 'g-')

        #print(str(n_inter) + " eu Lagrange(eq): " + str(np.linalg.norm(np.subtract(y2, y1))))
        #print(str(n_inter) + " max Lagrange(eq): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))
        #plt.show()


        # Hermite

        f_show_points(x_cheby, 'k.')
        x_draw = np.linspace(beg, end, n_draw)
        y2 = hermite2(f, df, x_cheby, x_draw)
        print(str(n_inter) + " eu Hermite(eq): " + str(np.linalg.norm(np.subtract(y2, y1))))
        print(str(n_inter) + " max Hemrite(eq): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))
        res.append([i, np.linalg.norm(np.subtract(y2, y1)), np.linalg.norm(np.subtract(y2, y1), np.inf)])
    save('results2', res)


if __name__ == "__main__":
    sys.exit(main())