import numpy as np

data = np.loadtxt("nodes.txt")

x = data[:,0]
y = data[:,1]
t = data[:,2]

n = len(t)

print("Количество узлов:", n)

# Функция вычисления коэффициентов сплайна

def cubic_spline_coeffs(t, f):

    n = len(t)
    h = np.diff(t)

    A = np.zeros((n,n))
    b = np.zeros(n)

    # Natural boundary
    A[0,0] = 1
    A[n-1,n-1] = 1

    for i in range(1,n-1):

        A[i,i-1] = h[i-1]
        A[i,i] = 2*(h[i-1]+h[i])
        A[i,i+1] = h[i]

        b[i] = 3*((f[i+1]-f[i])/h[i] - (f[i]-f[i-1])/h[i-1])

    # Решение СЛАУ
    c = np.linalg.solve(A,b)

    # коэффициенты
    a0 = f[:-1]
    a1 = np.zeros(n-1)
    a2 = c[:-1]
    a3 = np.zeros(n-1)

    for j in range(n-1):

        a1[j] = (f[j+1]-f[j])/h[j] - h[j]*(2*c[j]+c[j+1])/3
        a3[j] = (c[j+1]-c[j])/(3*h[j])

    return a0,a1,a2,a3

# коэффициенты для x(t)
ax0,ax1,ax2,ax3 = cubic_spline_coeffs(t,x)


# коэффициенты для y(t)
ay0,ay1,ay2,ay3 = cubic_spline_coeffs(t,y)



# объединяем коэффициенты

coeffs_x = np.column_stack((ax0,ax1,ax2,ax3))
coeffs_y = np.column_stack((ay0,ay1,ay2,ay3))

print("Размер матрицы коэффициентов x:", coeffs_x.shape)
print("Размер матрицы коэффициентов y:", coeffs_y.shape)


# сохраняем
np.savetxt("coeffs_x.txt", coeffs_x,
           header="a0 a1 a2 a3",
           comments='')

np.savetxt("coeffs_y.txt", coeffs_y,
           header="b0 b1 b2 b3",
           comments='')

print("Файлы coeffs_x.txt и coeffs_y.txt сохранены")