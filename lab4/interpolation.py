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





def plot_hermite(x, y, dens):
    def calc_hermite(x1, x2, y1, y2, t1, t2):
        lod = dens // (len(x)-1)
        xnew = []
        ynew = []
        for i in range(lod):
            t = i / (lod - 1)
            b1 = 2 * t * t * t - 3 * t * t + 1
            b2 = -2 * t * t * t + 3 * t * t
            bt1 = t * t * t - 2 * t * t + t
            bt2 = t * t * t - t * t
            xnew.append(b1 * x1 + b2 * x2 + bt1 * t1[0] + bt2 * t2[0])
            ynew.append(b1 * y1 + b2 * y2 + bt1 * t1[1] + bt2 * t2[1])
        return xnew, ynew

    t1 = (x[1]-x[0])/2, (y[1]-y[0])/2
    t2 = (x[2]-x[0])/2, (y[2]-y[0])/2
    xnew, ynew = calc_hermite(x[0], x[1], y[0], y[1], t1, t2)
    for j in range(1, len(x)-2):
        x1, x2, x3, x4 = x[j-1], x[j], x[j+1], x[j+2]
        y1, y2, y3, y4 = y[j-1], y[j], y[j+1], y[j+2]
        t1 = (x3-x1)/2, (y3-y1)/2
        t2 = (x4-x2)/2, (y4-y2)/2
        auxx, auxy = calc_hermite(x2, x3, y2, y3, t1, t2)
        for i in auxx:
            xnew.append(i)
        for i in auxy:
            ynew.append(i)
    l = len(x)-1
    t1 = (x[l]-x[l-2])/2, (y[l]-y[l-2])/2
    t2 = (x[l]-x[l-1])/2, (y[l]-y[l-1])/2
    auxx, auxy = calc_hermite(x[l-1], x[l], y[l-1], y[l], t1, t2)
    for i in auxx:
        xnew.append(i)
    for i in auxy:
        ynew.append(i)


    plt.plot(xnew, ynew, 'k-')
    plt.plot(x, y, 'x')
    plt.show()
    return ynew


def f(x):
    return pow(x,2) - (10 * np.cos(np.pi * x))


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

    n_draw = 1000
    beg = -zak
    end = zak



    for i in range(3,20,2):
        y1 = show_fun(f, beg, end, n_draw, 'b-')

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
        #y2 = show_fun(lambda ar: intrepolate_lagrange(x_eq, y_eq, ar), beg, end, n_draw, 'y-')

        #print(str(n_inter) + " eu Lagrange(eq): " + str(np.linalg.norm(np.subtract(y2, y1))))
        #print(str(n_inter) + " max Lagrange(eq): " + str(np.linalg.norm(np.subtract(y2, y1), np.inf)))


        #plt.show()

        # Hermite

        #f_show_points(x_eq, 'c.')
        #y2 = show_fun(lambda ar: interpolate_hermit(x_eq, y_eq, ar), beg, end, n_draw, 'c-')

        y2 = plot_hermite(x_eq, y_eq, n_draw)

        mini = min(len(y2), len(y1))

        print(str(n_inter) + " eu Hermite(eq): " + str(np.linalg.norm(np.subtract(y2[:mini], y1[:mini]))))
        print(str(n_inter) + " max Hemrite(eq): " + str(np.linalg.norm(np.subtract(y2[:mini], y1[:mini]), np.inf)))







if __name__ == "__main__":
    sys.exit(main())