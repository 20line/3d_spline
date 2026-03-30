import numpy as np

# загрузка данных
nodes = np.loadtxt("nodes.txt")

x = nodes[:,0]
y = nodes[:,1]
t = nodes[:,2]

coeffs_x = np.loadtxt("coeffs_x.txt")
coeffs_y = np.loadtxt("coeffs_y.txt")

n = len(t)

# функция сплайна
def spline_value(t_val, t, coeffs):

    j = np.searchsorted(t, t_val) - 1

    if j < 0:
        j = 0
    if j >= len(coeffs):
        j = len(coeffs) - 1

    dt = t_val - t[j]

    a0,a1,a2,a3 = coeffs[j]

    return a0 + a1*dt + a2*dt**2 + a3*dt**3


# вычисление ошибок
rho = []

for i in range(n):

    tx = spline_value(t[i], t, coeffs_x)
    ty = spline_value(t[i], t, coeffs_y)

    dist = np.sqrt((tx - x[i])**2 + (ty - y[i])**2)

    rho.append(dist)

rho = np.array(rho)

# ======================
# статистика
# ======================

mean_error = np.mean(rho)
std_error = np.std(rho)

print("Средняя ошибка:", mean_error)
print("Стандартное отклонение:", std_error)