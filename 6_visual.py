import numpy as np
import matplotlib.pyplot as plt


def load_points(filename="contours.txt"):
    points = np.loadtxt(filename)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Файл contours.txt должен содержать 2 столбца: x y")

    return points


def thin_points(points, M=10):
    N = len(points)
    N_hat = N // M
    return points[::M][:N_hat]


def build_natural_spline_system(t, f):
    n = len(t)
    h = np.diff(t)

    A = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    for j in range(1, n - 1):
        A[j, j - 1] = h[j - 1]
        A[j, j] = 2.0 * (h[j - 1] + h[j])
        A[j, j + 1] = h[j]

        rhs[j] = 6.0 * (
            (f[j + 1] - f[j]) / h[j]
            - (f[j] - f[j - 1]) / h[j - 1]
        )

    return A, rhs


def compute_spline_coefficients(t, f):
    A, rhs = build_natural_spline_system(t, f)
    m = np.linalg.solve(A, rhs)

    n = len(t)
    h = np.diff(t)
    coeffs = np.zeros((n - 1, 4), dtype=float)

    for j in range(n - 1):
        c0 = f[j]
        c1 = (f[j + 1] - f[j]) / h[j] - h[j] * (2.0 * m[j] + m[j + 1]) / 6.0
        c2 = m[j] / 2.0
        c3 = (m[j + 1] - m[j]) / (6.0 * h[j])
        coeffs[j] = [c0, c1, c2, c3]

    return coeffs


def evaluate_piecewise_spline(t_nodes, coeffs, t_eval):
    values = np.zeros_like(t_eval, dtype=float)
    n = len(t_nodes)

    for j in range(n - 1):
        if j < n - 2:
            mask = (t_eval >= t_nodes[j]) & (t_eval < t_nodes[j + 1])
        else:
            mask = (t_eval >= t_nodes[j]) & (t_eval <= t_nodes[j + 1])

        dt = t_eval[mask] - t_nodes[j]
        c0, c1, c2, c3 = coeffs[j]

        values[mask] = c0 + c1 * dt + c2 * dt**2 + c3 * dt**3

    return values


def main():
    M = 10
    h_plot = 0.1

    points = load_points("contour.txt")
    nodes = thin_points(points, M=M)

    x_hat = nodes[:, 0]
    y_hat = nodes[:, 1]

    N_hat = len(nodes)
    t_nodes = np.arange(N_hat, dtype=float) * M
    t_end = t_nodes[-1]

    a_coeffs = compute_spline_coefficients(t_nodes, x_hat)
    b_coeffs = compute_spline_coefficients(t_nodes, y_hat)

    t_dense = np.arange(0.0, t_end + h_plot, h_plot)

    x_spline = evaluate_piecewise_spline(t_nodes, a_coeffs, t_dense)
    y_spline = evaluate_piecewise_spline(t_nodes, b_coeffs, t_dense)

    print(f"Количество узлов N̂ = {len(nodes)}")
    print(f"Параметр прореживания M = {M}")
    print(f"Диапазон параметра: [0, {t_end}]")
    print(f"Шаг визуализации h = {h_plot}")

    plt.figure(figsize=(9, 9))
    plt.plot(x_spline, y_spline, linewidth=2, label="Параметрический кубический сплайн")
    plt.scatter(x_hat, y_hat, s=8, alpha=0.8, color="red", label="Узловые точки P̂")

    plt.title("Параметрический кубический сплайн и узловые точки")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
