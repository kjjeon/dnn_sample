import numpy as np
import matplotlib.pyplot as plt

vector_list = []
num_point = 1000
for i in range(num_point):
    x = np.random.normal(0.0,0.50)
    y = 0.1 * x + 0.3 + np.random.normal(0.0,0.03)
    vector_list.append([x,y])

x_data = [v[0] for v in vector_list]
y_data = [v[1] for v in vector_list]

plt.plot(x_data,y_data,'ro')
plt.show()