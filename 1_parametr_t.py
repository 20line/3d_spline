import numpy as np

# Загружаем координаты из файла
data = np.loadtxt('contour.txt')
x = data[:, 0]
y = data[:, 1]

# Вычисляем расстояния между последовательными точками
dx = np.diff(x)
dy = np.diff(y)
distances = np.sqrt(dx**2 + dy**2)

# Вычисляем накопленную сумму расстояний (параметр t)
t = np.zeros(len(x))
t[1:] = np.cumsum(distances)

# Выводим массив t (можно проверить первые несколько значений)
print("Параметр t для каждой точки:")
print(t)

# При желании сохраняем в файл (например, вместе с координатами)
# Создаём матрицу: [x, y, t]
result = np.column_stack((x, y, t))
np.savetxt('contour_with_t.txt', result, header='x y t', comments='')

print("Файл contour_with_t.txt сохранён (содержит x, y, t).")