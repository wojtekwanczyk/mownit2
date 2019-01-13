import sys
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
    df.to_excel(filename, sheet_name='sheet1', index=False, header=False)


def f(x, y):
    k = 2
    m = 3
    #return (x-y)/(x+y)
    return k * m * y * np.sin(m * x) + k**2 * m * np.sin(m * x) * np.cos(m * x)


def f_real(x):
    return np.exp(-2 * np.cos(3 * x)) - 2 * np.cos(3 * x) + 1


def f2_real(x):
    return -1 * np.sin(4 * x) + x


def euler(x, y, h, t):
    #print("  x        y")
    xs = [x]
    ys = [y]
    #print('%.3f' % x + '    ' + '%.3f' % y)
    while x <= t - 0.00001:
        k = h * f(x, y)
        y += k
        x += h
        #print('%.3f' % x + '    ' + '%.3f' % y)
        xs.append(x)
        ys.append(y)
    return xs, ys


def r_k(x, y, h, t, degree):
    #print("  x        y")
    xs = [x]
    ys = [y]
    if degree == 2:
        while x <= t - 0.00001:
            m1 = h * f(x, y)
            y = y + h * f(x + h/2, y + m1/2)
            x = x + h
            xs.append(x)
            ys.append(y)
        return xs, ys
    else:
        while x <= t - 0.00001:
            m1 = f(x, y)
            m2 = f((x+h/2), (y+m1*h/2))
            m3 = f((x+h/2), (y+m2*h/2))
            m4 = f((x+h), (y+m3*h))
            m = ((m1 + 2*m2 + 2*m3 + m4)/6)
            y = y + m * h
            x = x + h
            xs.append(x)
            ys.append(y)
        return xs, ys


# finite differential method
# n - number of sections
def fdm(x0, xn, y0, yn, n):
    xs = np.linspace(x0, xn, n+1)
    b = np.zeros(n-1)
    a = np.zeros((n-1, n-1))
    h = xs[1]-xs[0]

    middle = 16 * h**2 - 2
    a[0][0] = middle
    a[0][1] = 1
    a[n-2][n-3] = 1
    a[n-2][n-2] = middle
    b[0] = 16 * xs[1] * h**2 - y0
    b[n-2] = 16 * xs[n-1] * h**2 - yn
    for i in range(1, n-2):
        a[i][i-1] = 1
        a[i][i] = middle
        a[i][i+1] = 1
        b[i] = 16 * xs[i+1] * h**2
    ys = np.linalg.solve(a, b)
    ys = list(ys)
    ys.insert(0, y0)
    ys.append(yn)
    return xs, ys


def draw(xs, ys, color, title):
    plt.plot(xs, ys, color, label=title, markersize=11)
    plt.draw()


def main():



    type = 2


    if type == 1:
        res = [['Krok metody', 'Błąd metody Eulera', 'Błąd metody RK2', 'Błąd metody RK4']]
        a = np.pi / 6
        b = 3 * np.pi / 2

        for n in range(4999, 5000, 200):
            h = (b-a) / (n-1)

            xs = np.linspace(a, b, n)
            ys = []
            for i in range(len(xs)):
                ys.append(f_real(xs[i]))
            draw(xs, ys, 'k.', 'Real function')

            xs, y2 = euler(a, f_real(a), h, b)
            ee = np.linalg.norm(np.subtract(y2, ys))
            draw(xs, y2, 'b.', 'Euler\'s method')

            xs, y2 = r_k(a, f_real(a), h, b, 2)
            erk2 = np.linalg.norm(np.subtract(y2, ys))
            draw(xs, y2, 'r.', 'RK 2nd degree method')

            xs, y2 = r_k(a, f_real(a), h, b, 4)
            erk4 = np.linalg.norm(np.subtract(y2, ys))
            draw(xs, y2, 'g.', 'RK 4th degree method')

            line = [h, ee, erk2, erk4]
            print(line)
            res.append(line)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid()
            plt.show()
    else:
        res = [['Krok metody', 'Błąd metody RS']]
        a = 0
        b = (2 * np.pi + 1) / 4

        xr = np.linspace(a, b, 100)
        yr = [f2_real(x) for x in xr]

        for n in range(10, 200, 10):
            print(n)
            h = (b - a) / (n - 1)

            xs = np.linspace(a, b, n+1)
            ys = []
            for i in range(len(xs)):
                ys.append(f2_real(xs[i]))
            draw(xr, yr, 'k', 'Real function')

            xs, y2 = fdm(a, b, 0, f2_real(b), n)
            er = np.linalg.norm(np.subtract(y2, ys))
            draw(xs, y2, 'm.', 'FDM method')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid()
            plt.show()

            line = [h, er]
            print(line)
            res.append(line)

    #print(res)
    #save('results5', res)


if __name__ == "__main__":
    sys.exit(main())
