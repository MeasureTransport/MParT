import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.dat")
x = data[:,0]
y_true = data[:,1]
y_measured = data[:,2]
map_of_x_before = data[:,3]
map_of_x_after = data[:,4]

plt.figure()
#plt.title('Starting map error: {:.2E} / Final map error: {:.2E}'.format(error_before, error_after))
plt.plot(x.flatten(),y_true.flatten(),'*--',label='true data', alpha=0.8)
plt.plot(x.flatten(),y_measured.flatten(),'*--',label='measured data', alpha=0.4)
plt.plot(x.flatten(),map_of_x_before.flatten(),'*--',label='initial map output', color="green", alpha=0.8)
plt.plot(x.flatten(),map_of_x_after.flatten(),'*--',label='final map output', color="red", alpha=0.8)
plt.legend()
plt.show()

