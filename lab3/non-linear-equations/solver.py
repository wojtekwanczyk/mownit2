import sys
import numpy as np
from pandas import DataFrame
from random import randint as rand
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# show precision
# np.set_printoptions(floatmode='unique')


# here change computation precision
type_object = np.float64()
my_type = type(type_object)


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
    df = df.transpose()
    df.to_excel(filename, sheet_name='sheet1', index=False, header=False)


def f(x):
    return (x-1) * (np.exp(1)**(-15*x)) + (x**13)


def f_der(x):
    return np.exp(1)**(-15*x) * (13 * (x**12) * np.exp(1)**(15*x) - 15*x + 16)


def print_polynomial(f):
    if f[-1] < 0:
        sys.stdout.wrtie('-')
    for i in range(len(f)-1, 0, -1):
        if f[i] != 0:
            sys.stdout.write(str(abs(f[i])) + 'x^' + str(i))
            if f[i-1] >= 0:
                sys.stdout.write(' + ')
            else:
                sys.stdout.write(' - ')
    print(abs(f[0]))


def get_value(f, x):
    res = 0
    for i in range(len(f)-1, -1, -1):
        res += f[i] * x**i
    return res


def derivative(f):
    f2 = np.full(len(f) - 1, 0, dtype=my_type)
    for i in range(len(f2)-1, -1, -1):
        f2[i] = f[i+1] * (i+1)
    return f2


# f - array z kolejnymi wspolczynnikami
def secant_method(a, b, ro, cond):

    iterations = 0
    if cond == 1:
        cond_val = abs(b - a)
    else:
        cond_val = abs(f(b))

    while cond_val >= ro:
        iterations += 1
        # print(a, b)

        x = b - ((f(b) * (b-a)) / (f(b) - f(a)))
        a, b = b, x

        if cond == 1:
            cond_val = abs(b - a)
        else:
            cond_val = abs(f(b))

    # print("Iterations: " + str(iterations))
    #print('cond: ' + str(cond) + ' x: ' + str(b))
    return b, iterations


def newton_method(x, ro, cond):

    iterations = 0
    if cond == 1:
        cond_val = 100
    else:
        cond_val = abs(f(x))

    while cond_val >= ro:
        iterations += 1
        # print(y)

        x_bkp = x
        x = x_bkp - (f(x_bkp) / f_der(x_bkp))

        if cond == 1:
            cond_val = abs(x - x_bkp)
        else:
            cond_val = abs(f(x))

    # print("Iterations: " + str(iterations))
    return x, iterations


def show(x):
    print("x: " + str(x) + '\tf(x): ' + str(f(x)))


def calc_1():
    a = -0.8
    b = 0.9
    ro = 1e-14
    cond = 1
    samples = 17
    step = 0.1

    results = [['newton', 'cond1', '', '']]
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    s = a
    for i in range(samples):
        x1, i1 = newton_method(s, 1e-3, cond)
        x2, i2 = newton_method(s, 1e-8, cond)
        x3, i3 = newton_method(s, 1e-15, cond)
        results.append([s] + [i1, i2, i3])
        #results.append([s] + [x1, x2, x3])
        s += step

    results.append(['', '', '', ''])
    results.append(['newton', 'cond2', '', ''])
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    cond = 2
    s = a
    for i in range(samples):
        x1, i1 = newton_method(s, 1e-3, cond)
        x2, i2 = newton_method(s, 1e-8, cond)
        x3, i3 = newton_method(s, 1e-15, cond)
        results.append([s] + [i1, i2, i3])
        #results.append([s] + [x1, x2, x3])
        s += step

    results.append(['', '', '', ''])
    results.append(['sieczne', 'cond1', 'od a', ''])
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    cond = 1
    s = a
    for i in range(samples):
        x1, i1 = secant_method(s, b, 1e-3, cond)
        x2, i2 = secant_method(s, b, 1e-8, cond)
        x3, i3 = secant_method(s, b, 1e-15, cond)
        results.append([s] + [i1, i2, i3])
        #results.append([s] + [x1, x2, x3])
        s += step

    results.append(['', '', '', ''])
    results.append(['sieczne', 'cond2', 'od a', ''])
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    cond = 2
    s = a
    for i in range(samples):
        x1, i1 = secant_method(s, b, 1e-3, cond)
        x2, i2 = secant_method(s, b, 1e-8, cond)
        x3, i3 = secant_method(s, b, 1e-15, cond)
        results.append([s] + [i1, i2, i3])
        #results.append([s] + [x1, x2, x3])
        s += step

    results.append(['', '', '', ''])
    results.append(['sieczne', 'cond1', 'od b', ''])
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    cond = 1
    s = b
    for i in range(samples):
        x1, i1 = secant_method(a, s, 1e-3, cond)
        x2, i2 = secant_method(a, s, 1e-8, cond)
        x3, i3 = secant_method(a, s, 1e-15, cond)
        results.append([s] + [i1, i2, i3])
        #results.append([s] + [x1, x2, x3])
        s -= step

    results.append(['', '', '', ''])
    results.append(['sieczne', 'cond2', 'od b', ''])
    results.append(['start \\ ro', 1e-3, 1e-8, 1e-15])
    cond = 2
    s = b
    for i in range(samples):
        x1, i1 = secant_method(a, s, 1e-3, cond)
        x2, i2 = secant_method(a, s, 1e-8, cond)
        x3, i3 = secant_method(a, s, 1e-15, cond)
        #results.append([s] + [x1, x2, x3])
        results.append([s] + [i1, i2, i3])
        s -= step

    save("wyniki3", results)


def f_3(x):
    ret = np.zeros(len(x))
    ret[0] = x[0]**2 + x[1]**2 - x[2]**2 - 1
    ret[1] = x[0] - 2 * x[1]**3 + 2 * x[2]**2 + 1
    ret[2] = 2 * x[0]**2 + x[1] - 2 * x[2]**2 - 1

    # print(ret)
    return ret


def mul2(x):
    return 2 * x


def mul_2(x):
    return -2 * x


def mul4(x):
    return 4 * x


def mul_4(x):
    return -4 * x


def mul_6_sqr(x):
    return -6 * (x**2)


def one(x):
    return 1


jacobian = [[mul2, mul2, mul_2],
            [one, mul_6_sqr, mul4],
            [mul4, one, mul_4]]


def solve(start, ro, cond):
    print(start)
    n = 3
    jacobian_instance = np.full((n, n), 0)

    if cond == 1:
        cond_val = 100
    else:
        cond_val = abs(np.linalg.norm(f_3(start), ord=np.inf))
    new = []

    while cond_val >= ro:
        for i in range(n):
            for j in range(n):
                jacobian_instance[i][j] = jacobian[i][j](start[j])

        new = start - np.linalg.inv(jacobian_instance).dot(f_3(start))
        #print(new)

        if cond == 1:
            cond_val = abs(np.linalg.norm(new - start, ord=np.inf))
        else:
            cond_val = abs(np.linalg.norm(f_3(new), ord=np.inf))

        start = new

    #print(new)
    return new


def test_and_add(i1, j1, k1, xs, ys, zs):
    eps = 0.1
    vec = f_3([i1, j1, k1])
    # print(vec)
    for p in range(3):
        if vec[p] == 0:
            xs[p].append(i1)
            ys[p].append(j1)
            zs[p].append(k1)
    if abs(vec[0]) < eps and abs(vec[1]) < eps and abs(vec[2]) < eps:
        print("FOUND: " + str(i1) + ' ' + str(j1)
              + ' ' + str(k1) + ' ' + str(vec))
    return xs, ys, zs




def main():
    cond = 2

    found = ['found']
    vectors = ['vectors']
    not_found = ['not found - overflow']
    not_found_2 = ['not found - singular matrix']


    il = 1
    zak = 10
    for i in range(100):
        start = [(rand(-zak, zak) * -il), (rand(-zak, zak) * il), (rand(-zak, zak) * il)]
        #print(start)
        try:
            v = solve(start, 1e-1, cond)
            print('Found: ' + str(start) + ' !!!!!!!!')
            found.append(start)
            vectors.append(v)
        except OverflowError:
            print('Not found(overflow) ' + str(start))
            not_found.append(start)
        except np.linalg.linalg.LinAlgError:
            print('Not found(singular matrix) ' + str(start))
            not_found_2.append(start)

    res = [found, vectors, not_found, not_found_2]
    save('wyniki4', res)




    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    n = 50
    il = 1
    xs = [[], [], []]
    ys = [[], [], []]
    zs = [[], [], []]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                i1 = i * il
                j1 = j * il
                k1 = k * il
                xs, ys, zs = test_and_add(i1, j1, k1, xs, ys, zs)
                xs, ys, zs = test_and_add(i1, j1, -k1, xs, ys, zs)
                xs, ys, zs = test_and_add(i1, -j1, k1, xs, ys, zs)
                xs, ys, zs = test_and_add(-i1, j1, k1, xs, ys, zs)
                xs, ys, zs = test_and_add(-i1, -j1, k1, xs, ys, zs)
                xs, ys, zs = test_and_add(-i1, j1, -k1, xs, ys, zs)
                xs, ys, zs = test_and_add(i1, -j1, -k1, xs, ys, zs)
                xs, ys, zs = test_and_add(-i1, -j1, -k1, xs, ys, zs)


    #ax.scatter3D(xs[0], ys[0], zs[0], 'gray')
    #ax.scatter3D(xs[1], ys[1], zs[1], 'blue')
    ax.scatter(xs[2], ys[2], zs[2])
    
    plt.show()

    '''




if __name__ == "__main__":
    sys.exit(main())
