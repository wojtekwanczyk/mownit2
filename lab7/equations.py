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


# y'' = u + vy + wy'
# ddf_eq = {'u': u(x), 'v': v(x), 'w': w(x)} \ - lambda
def finite_difference_method(x0, y0, xn, yn, n, ddf_eq):
    h = (xn - x0) / n

    x = np.linspace(x0, xn, n)
    a, b, d, c = np.zeros([n]), np.zeros([n]), np.zeros([n]), np.zeros([n])
    for i in range(1, n-1):
        a[i] = -(1 + ddf_eq['w'](x[i]) * h / 2)
        b[i] = -ddf_eq['u'](x[i]) * h**2
        c[i] = -(1 - ddf_eq['w'](x[i]) * h / 2)
        d[i] = (2 + ddf_eq['v'](x[i]) * h**2)

    A = np.zeros([n, n])
    for i in range(2, n-2):
        A[i][i] = d[i]
        A[i][i-1] = a[i]
        A[i][i+1] = c[i]
    A[1][1], A[n-2][n-2] = d[1], d[n-2]
    A[1][2] = c[1]
    A[n-2][n-3] = a[n-2]

    b[1] = b[1] - a[1] * y0
    b[n-2] = b[n-2] - c[n-2] * yn
    b[0], b[n-1] = y0, yn
    A[0][0] = 1
    A[n-1][n-1] = 1

    y = np.linalg.solve(A, b)   # Ay = b

    return x, y


def mrs(eq, a, b, alfa, beta, n):
    x = np.linspace(a, b, n+1)
    b = np.zeros(n+2)
    a = np.zeros((n+2, n+2))
    h = x[1]-x[0]

    b[0] = alfa
    for i in range(1, n+1):
        b[i] = h**2 * x[i]
    b[n+1] = beta
    print(b)

    a[0][0] = 1
    for i in range(1, n+1):
        a[i][i-1] = 1
        a[i][i] = -2
        a[i][i+1] = 1
    a[n+1][n+1] = beta

    y = np.linalg.solve(a, b)
    return x, y


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

    n = int((b-a) / h + 1)

    xs = np.linspace(a, b, n)
    ys = []
    for i in range(len(xs)):
        ys.append(f2_real(xs[i]))
    draw(xs, ys, 'k.', 'Real function')

    # y'' = u + vy + wy'
    ddf_eq = {'u': lambda x: 16 * x, 'v': lambda x: -16 * x, 'w': lambda x: 0}

    xs, ys = finite_difference_method(0, 0, b, f2_real(b), n, ddf_eq)
    draw(xs, ys, 'g.', 'FDM method')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.suptitle("Big Title")
    plt.grid()
    plt.show()

    #xs, ys = mrs(ddf_eq, 0, 2, 2, 1, 4)

    #print(len(xs), len(ys))
    #draw(xs, ys, 'm.', 'FDM method')



if __name__ == "__main__":
    sys.exit(main())
