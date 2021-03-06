import multiprocessing
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt


# show precision
np.set_printoptions(floatmode='unique')


# here change computation precision
type_object = np.float64()
my_type = type(type_object)


# wartosci własne - eigen values
def spectral_radius(a):
    eigvals = np.linalg.eigvals(a)
    # for i in range(eigvals.size):
        # print(str(i+1) + ' wartosc wlasna: ' + str(eigvals[i]))
    return abs(max(eigvals, key=abs))


def gen_x(n):
    x = np.ones(n, dtype=my_type)
    for i in range(n):
        if i % 2 == 0:
            x[i] = -x[i]
    return x


def gen_a(n):
    # podpunkt b
    k = 10
    m = 5
    a = np.zeros((n, n), dtype=my_type)
    for i in range(n):
        for j in range(n):
            if i == j:
                a[i][j] = k
            else:
                a[i][j] = 1/(abs(i - j) + m)
    return a


def jacobi_method(a, b, x, n, ro, nr, stop_cond=1):
    # set start vector
    x1 = np.zeros(n, dtype=my_type)

    for ind, elem in enumerate(x1):
        x1[ind] = x[ind] + (random.randint(0, 9) * 1e-8)

    x2 = np.full(n, 0, dtype=my_type)
    xdiff = np.full(n, 100, dtype=my_type)
    m = np.full((n, n), 0, dtype=my_type)

    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = -(a[i][j] * 1/a[i][i])

    print("Spectral radius: " + str(spectral_radius(m)))

    if stop_cond == 1:
        cond = np.linalg.norm(xdiff - x1)
    else:
        cond = np.linalg.norm(np.dot(a, x1) - b)

    k = 0
    while cond > ro:
        k += 1
        if k % 1000 == 0:
            print(str(time.process_time()) + ': nr ' + str(nr) + ' iteracja ' + str(k))
        for i in range(n):
            x2[i] = (1/a[i][i]) * b[i]
            for j in range(n):
                x2[i] += m[i][j] * x1[j]

        xdiff = x1.copy()
        x1 = x2.copy()

        if stop_cond == 1:
            cond = np.linalg.norm(xdiff - x1)
        else:
            cond = np.linalg.norm(np.dot(a, x1) - b)

    print('Number of iterations: ' + str(k))

    return x1, k


def sor_method(a, b, x, n, ro, nr, stop_cond, w):
    # set start vector
    x1 = np.full(n, 0, dtype=my_type)
    x2 = np.full(n, 100, dtype=my_type)

    # for ind, elem in enumerate(x1):
    #   x1[ind] = x[ind] + (random.randint(0, 9) * 1e-8)


    if stop_cond == 1:
        cond = np.linalg.norm(x2 - x1)
    else:
        cond = np.linalg.norm(np.dot(a, x2) - b)

    k = 0
    #w = 1.5          # (1, 2) - dla relaksacji
    while cond > ro:
        # print('Norm: ' + str(cond))
        k += 1
        if k % 1000 == 0:
            print(str(time.process_time()) + ': nr ' + str(nr) + ' iteracja ' + str(k))

        for i in range(n):
            tmp = b[i]
            for j in range(i):
                tmp -= a[i][j] * x2[j]
            for j in range(i+1,n):
                tmp -= a[i][j] * x1[j]
            tmp = w/a[i][i] * tmp
            x2[i] = (1-w) * x1[i] + tmp

        if stop_cond == 1:
            cond = np.linalg.norm(x2 - x1)
        else:
            cond = np.linalg.norm(np.dot(a, x2) - b)
        x1 = x2.copy()

    print('Number of iterations: ' + str(k))

    return x1, k


def print_input(a, x, b):
    print('a')
    print(a)
    print('----------\nx')
    print(x)
    print('----------\nb')
    print(b)


def new_jacobi(n, ro, stop_cond, result, nr):
    a = gen_a(n)
    x = gen_x(n)
    b = np.dot(a, x)

    print('Type set to: ' + str(my_type))
    print('RO = ' + str(ro))

    start_time = time.process_time()
    x_solution, iterations = jacobi_method(a, b, x, n, ro, nr, stop_cond)
    end_time = time.process_time()

    elapsed_time = end_time - start_time
    x_diff = np.linalg.norm(x_solution - x)

    res = (nr, n, ro, x_diff, elapsed_time, iterations)
    print(res)

    result.append(res)


def main():

    #print(format(x, '.20f'))
    #print(format(y, '.20f'))
    #print(type(x_solution[1]))



    manager = multiprocessing.Manager()
    result = manager.list()
    jobs = []

    size = 1000
    stop_cond = 1

    process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, stop_cond, result, 1))
    jobs.append(process)
    process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, 2, result, 2))
    jobs.append(process)

    #process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, stop_cond, result, 3))
    #jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    print('WYNIKI')
    for item in result:
        print(item)

    '''
    process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, stop_cond, result, 1, 1.1))
    jobs.append(process)
    process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, stop_cond, result, 1, 1.5))
    jobs.append(process)
    process = multiprocessing.Process(target=new_jacobi, args=(size, 1e-12, stop_cond, result, 1, 1))
    jobs.append(process)
    '''


    '''
    n = 200
    a = gen_a(n)
    x = gen_x(n)
    b = np.dot(a, x)

    xs = []
    iter_list = []
    errors = []
    w = 0.1
    count = 18
    for i in range(count):
        print(w)
        x1, itera = sor_method(a, b, x, n, 1e-12, i, stop_cond, w)
        xs.append(w)
        errors.append(np.linalg.norm(x1 - x))
        w += 1.8/(count-1)
        iter_list.append(itera)




    print('WYNIKI')
    for i, item in enumerate(xs):
        print(str(item) + '\t' + str(errors[i]))

    plt.plot(xs, errors, '.')
    plt.xlabel('Wartość współczynnika relaksacji')
    plt.ylabel('Norma maximum z różnicy między wektorem otrzymanym i oczekiwanym')
    plt.grid(True)
    plt.show()
    '''

    return 0


if __name__ == "__main__":
    sys.exit(main())
