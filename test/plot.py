import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('data/TEK16.txt', delimiter='\n', unpack=True)
print (x)
print(x.shape)
plt.plot(x, label='Loaded from file!')

plt.show()