import sys
import numpy as np
import xlwt
from pandas import DataFrame
import openpyxl
import matplotlib.pyplot as plot


# show precision
# np.set_printoptions(floatmode='unique')


# here change computation precision
type_object = np.float64()
my_type = type(type_object)


def save(filename, results):
    filename += '.xlsx'
    df = DataFrame(data=results)
    # print(df)
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


def main():
    # f = [3, 0, 0, 3, -2]
    # f.reverse()
    # print_polynomial(f)
    



if __name__ == "__main__":
    sys.exit(main())
