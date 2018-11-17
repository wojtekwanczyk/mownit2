import numpy as np
import matplotlib.pyplot as plt


def get_a(n):
    a = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                a[i,j] = 6
            elif i+1 == j:
                a[i,j] = 1/(i + 1 + 2)
            elif i-1 == j:
                a[i,j] = 6/(i + 1 + 6 + 1)
    return a


'''
n = 1000
                
b = np.zeros((n,n))
b.shape = (n, n)

for i in range(0, n):
    for j in range(0, n):
        if(j >= i):
            b[i,j] = (2*(i+1)) / (j+1)
        else:
            b[i,j] = b[j,i]
'''
# print(a)
# print('a: ' + str(np.linalg.norm(a) * np.linalg.norm(np.linalg.inv(a))))
# print(b)
# print('b: ' + str(np.linalg.norm(b) * np.linalg.norm(np.linalg.inv(b))))


xs = []
ys = []

size = 200
for i in range(20):
    a = get_a(size)
    wsk = np.linalg.norm(a) * np.linalg.norm(np.linalg.inv(a))
    xs.append(size)
    ys.append(wsk)
    print(str(size) + ' size: ' + str(wsk))
    size += 200

plt.plot(xs, ys, '--', xs, ys, '.r', markersize=10)
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Uwarunkowanie macierzy')
plt.grid(True)
plt.show()
