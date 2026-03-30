import numpy as np

def lab1_base(filename_in: str, factor: int, filename_out: str):
    # 1. Загрузка исходных точек
    data = np.loadtxt(filename_in)
    x = data[:, 0]
    y = data[:, 1]
    N = len(x)
    print(f"Загружено точек: {N}")

    # 2. Вычисление параметра t (накопленная длина дуги)
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    t = np.zeros(N)
    t[1:] = np.cumsum(dist)
    print(f"Параметр t от {t[0]:.3f} до {t[-1]:.3f}")

    # 3. Прореживание
    N_hat = N // factor
    indices = np.arange(0, N_hat * factor, factor)
    t_hat = t[indices]
    x_hat = x[indices]
    y_hat = y[indices]
    print(f"Узлов после прореживания: {len(x_hat)}")

    # 4. Построение сплайна для одной координаты (возвращает коэффициенты a0,a1,a2,a3)
    def cubic_spline_coeffs(t_nodes, f_values):
        n = len(t_nodes)
        h = np.diff(t_nodes)

        # Формируем трёхдиагональную систему для вторых производных
        A = np.zeros((n, n))
        b = np.zeros(n)

        # Граничные условия (естественный сплайн)
        A[0, 0] = 1
        A[-1, -1] = 1

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i]     = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]

            # Правая часть с коэффициентом 6 (для вторых производных)
            b[i] = 6 * ((f_values[i + 1] - f_values[i]) / h[i] -
                        (f_values[i] - f_values[i - 1]) / h[i - 1])

        # Решение системы
        c = np.linalg.solve(A, b)   # вторые производные в узлах

        # Вычисление коэффициентов сплайна на каждом интервале
        a0 = f_values[:-1]
        a1 = np.zeros(n - 1)
        a2 = c[:-1]
        a3 = np.zeros(n - 1)

        for j in range(n - 1):
            a1[j] = (f_values[j + 1] - f_values[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 6
            a3[j] = (c[j + 1] - c[j]) / (6 * h[j])

        return a0, a1, a2, a3

    # Строим сплайны для x и y
    ax0, ax1, ax2, ax3 = cubic_spline_coeffs(t_hat, x_hat)
    ay0, ay1, ay2, ay3 = cubic_spline_coeffs(t_hat, y_hat)

    # 5. Объединение коэффициентов в матрицу (N̂-1)×8
    n_intervals = len(ax0)
    coeffs = np.zeros((n_intervals, 8))
    coeffs[:, 0] = ax0
    coeffs[:, 1] = ax1
    coeffs[:, 2] = ax2
    coeffs[:, 3] = ax3
    coeffs[:, 4] = ay0
    coeffs[:, 5] = ay1
    coeffs[:, 6] = ay2
    coeffs[:, 7] = ay3

    # 6. Сохранение в файл
    np.savetxt(filename_out, coeffs,
               header="a0 a1 a2 a3 b0 b1 b2 b3", comments='')
    print(f"Коэффициенты сохранены в {filename_out} (размер {coeffs.shape})")

    # Возвращать ничего не требуется, но можно вернуть coeffs, если нужно
    return coeffs


# Пример использования (можно закомментировать при импорте)
if __name__ == "__main__":
    # Замените "contour.txt" на ваш файл с исходными точками
    lab1_base("contour.txt", factor=10, filename_out="coeffs.txt")