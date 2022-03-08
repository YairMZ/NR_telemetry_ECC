import numpy as np
import matplotlib.pyplot as plt

c = lambda p: 1 + (p*np.log2(p) + (1-p)*np.log2(1-p))
f = np.linspace(0.03, 0.1, num=50)
capacity = np.array([c(p) for p in f])

r = 0.24  # percent of bits decoded by NE decoder
nr_capacity = np.array([(1-r)*c(p) + r*c(p/2) for p in f])
plt.plot(f, capacity, f, nr_capacity, f, np.array([2/3] * 50))
plt.xlabel("BSC bit flip probability p")
plt.ylabel("Channel capacity / Maximal code rate for reliable communication")
plt.legend(['BSC capacity', 'modified capacity', 'simulation rate'])
plt.show()
