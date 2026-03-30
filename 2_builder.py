import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('contour.txt')

x = data[:, 0]
y = data[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=1, color='blue', label='исходные точки')
plt.axis('equal')
plt.title('Точки контура')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()