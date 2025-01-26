import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 400)

y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

plt.plot(x, y1, label='y = 2x + 1', color='black', linestyle='--')
plt.plot(x, y2, label='y = 2x + 2', color='gray', linestyle='-.')
plt.plot(x, y3, label='y = 2x + 3', color='black', linestyle=':')

# Add title and labels
plt.title('Graphs of y = 2x + 1, y = 2x + 2, y = 2x + 3')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
