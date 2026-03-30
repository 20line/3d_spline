import numpy as np
import matplotlib.pyplot as plt

# Загрузка исходных точек
data = np.loadtxt('contour.txt')
x = data[:, 0]
y = data[:, 1]

# Вычисление параметра t (накопленная длина дуги)
dx = np.diff(x)
dy = np.diff(y)
dist = np.sqrt(dx**2 + dy**2)
t = np.zeros(len(x))
t[1:] = np.cumsum(dist)

# Параметр прореживания
M = 10
N = len(x)
N_hat = N // M                     # количество узлов
indices = np.arange(0, N_hat * M, M)  # индексы узлов

# Узлы интерполяции
x_hat = x[indices]
y_hat = y[indices]
t_hat = t[indices]

print(f"Исходных точек: {N}")
print(f"Узлов (каждая {M}-я точка): {len(x_hat)}")

# Визуализация
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=0.5, label='исходный контур')
plt.scatter(x_hat, y_hat, c='red', s=30, label=f'узлы (M={M})')
plt.axis('equal')
plt.title('Прореженные узлы интерполяции')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Сохранение узлов с параметром t
np.savetxt('nodes.txt', np.column_stack((x_hat, y_hat, t_hat)),
           header='x y t', comments='')
print("Файл nodes.txt сохранён (содержит x, y, t для узлов).")