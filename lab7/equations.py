import sys
import matplotlib.pyplot as plt
import numpy as np


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
    print("  x        y")
    xs = [x]
    ys = [y]
    while x <= t - 0.00001:
        k = h * f(x, y)
        y += k
        x += h
        #print('%.3f' % x + '    ' + '%.3f' % y)
        xs.append(x)
        ys.append(y)
    return xs, ys


def r_k(x, y, h, t):
    print("  x        y")
    xs = [x]
    ys = [y]
    while x <= t - 0.00001:
        m1 = f(x, y)
        m2 = f((x+h/2), (y+m1*h/2))
        m3 = f((x+h/2), (y+m2*h/2))
        m4 = f((x+h), (y+m3*h))
        m = ((m1 + 2*m2 + 2*m3 + m4)/6)
        y = y + m * h
        x = x + h
        #print('%.3f' % x + '    ' + '%.3f' % y)
        xs.append(x)
        ys.append(y)
    return xs, ys


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
    plt.plot(xs, ys, color, label=title, markersize=5)
    plt.draw()


def main():

    '''
    a = np.pi / 6
    b = 3 * np.pi / 2
    h = 0.1

    n = int((b-a) / h + 1)
    print(n)

    xs = np.linspace(a, b, n)
    ys = []
    for i in range(len(xs)):
        ys.append(f_real(xs[i]))
    draw(xs, ys, 'k.', 'Real function')

    xs, ys = euler(a, f_real(a), h, b)
    draw(xs, ys, 'b.', 'Euler\'s method')

    xs, ys = r_k(a, f_real(a), h, b)
    draw(xs, ys, 'r.', 'Runge-Kutta method')
    '''

    a = 0
    b = (2 * np.pi + 1) / 4
    h = 0.01

    n = int((b-a) / h)
    print(n)

    xs = np.linspace(a, b, n)
    ys = []
    for i in range(len(xs)):
        ys.append(f2_real(xs[i]))
    draw(xs, ys, 'k.', 'Real function')

    xs, ys = fdm(a, b, 0, f2_real(b), n)
    draw(xs, ys, 'm.', 'FDM method')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.suptitle("Big Title")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
