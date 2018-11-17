import sys
import numpy as np


# show precision
np.set_printoptions(floatmode='unique')
ro = 1e-3
  

# here change computation precision
type_object = np.float64()
my_type = type(type_object)

def f(x):
    return (x-1) * (np.exp(1)**(-15*x) + x**13)


def f_der(x):
    return np.exp(1)**(-15*x) * (13 * x**12 * np.exp(1)**(15*x) - 15*x + 16)



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
def secant_method(a, b):
    x = 0
    i = 0
    max_iter = 100

    cond = 1

    if cond == 1:
        y = f(x)
    else:
        y = 100
    while a != b and i < max_iter:
        # print(a, b)
        #x = b - ((get_value(f, b) * (b - a)) / (get_value(f, b) - get_value(f, a)))
        x = b - ((f(b) * (b-a)) / (f(b) - f(a)))

        a, b = b, x
        i += 1
        x = x - (f(x) / f_der(x))
        if cond == 1:
            y = f(x)
        else:
            y = abs(x - b)

    if i == max_iter:
        return np.nan

    return x


def newton_method(x):
    max_iter = 100
    i = 0
    #f2 = derivative(f)

    cond = 2

    if cond == 1:
        y = f(x)
    else:
        y = 100

    while abs(y) >= ro and i < max_iter:
        # print(y)
        #x = x - (y / get_value(f2, x))
        x_bkp = x
        x = x - (f(x) / f_der(x))
        if cond == 1:
            y = f(x)
        else:
            y = abs(x_bkp - x)
        i += 1

    print("Iterations: " + str(i))
    if i == max_iter:
        return np.nan

    return x




def main():
    #f = [3, 0, 0, 3, -2]
    #f.reverse()
    #print_polynomial(f)

    # res1 = secant_method(f, -100, 0)
    # res2 = newton_method(f, 0)
    # print('Secant: ' + str(res1) + '\nNewton: ' + str(res2))

    s = -0.8
    for i in range(18):
        res = newton_method(s)
        print("Start: " + str(s) + "\t\tValue: " + str(res))
        # print(np.exp(1))
        s += 0.1

    a = -0.8
    b = 0.9

    s = a
    for i in range(18):
        res = secant_method(s, b)
        print("Start: " + str(s) + "\tEnd: " + str(b) + "\t\tValue: " + str(res))
        # print(np.exp(1))
        s += 0.1
    
    s = b
    for i in range(18):
        res = secant_method(a, s)
        print("Start: " + str(a) + "\tEnd: " + str(s) + "\t\tValue: " + str(res))
        # print(np.exp(1))
        s -= 0.1


if __name__ == "__main__":
    sys.exit(main())