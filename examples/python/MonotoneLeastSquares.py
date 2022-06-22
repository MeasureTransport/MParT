from mpart import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# geometry
num_points = 1000
xmin, xmax = 0, 4
x = np.linspace(xmin, xmax, num_points)[None,:]


# measurements
noisesd = 0.4

# note: data might not monotone because of the noise, 
# but we assume the true underlying function is monotone.
y_true = 2*(x>2)
y_noise = noisesd*np.random.randn(num_points) 
y_measured = y_true + y_noise

# Create multi-index set:
multis = np.array([[0], [1], [2], [3], [4], [5]])
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# Least squares objective
def objective(coeffs, monotoneMap, x, y_measured):
    monotoneMap.SetCoeffs(coeffs)
    map_of_x = monotoneMap.Evaluate(x)
    return np.sum((map_of_x - y_measured)**2)/x.shape[1]

# Before optimization
map_of_x_before = monotoneMap.Evaluate(x)
error_before = objective(monotoneMap.CoeffMap(), monotoneMap, x, y_measured)

# Optimize
res = minimize(objective, monotoneMap.CoeffMap(), args=(monotoneMap, x, y_measured), method="Nelder-Mead")

# After optimization
map_of_x_after = monotoneMap.Evaluate(x)
error_after = objective(monotoneMap.CoeffMap(), monotoneMap, x, y_measured)

# Plot data (before and after together)
plt.figure()
plt.title('Starting map error: {:.2E} / Final map error: {:.2E}'.format(error_before, error_after))
plt.plot(x.flatten(),y_true.flatten(),'*--',label='true data', alpha=0.8)
plt.plot(x.flatten(),y_measured.flatten(),'*--',label='measured data', alpha=0.4)
plt.plot(x.flatten(),map_of_x_before.flatten(),'*--',label='initial map output', color="green", alpha=0.8)
plt.plot(x.flatten(),map_of_x_after.flatten(),'*--',label='final map output', color="red", alpha=0.8)
plt.legend()
plt.show()


# Plot data (before and after apart)
plt.figure()
plt.title('Starting map error: {:.2E}'.format(error_before))
plt.plot(x.flatten(),y_true.flatten(),'*--',label='true data', alpha=0.8)
plt.plot(x.flatten(),y_measured.flatten(),'*--',label='measured data', alpha=0.4)
plt.plot(x.flatten(),map_of_x_before.flatten(),'*--',label='initial map output', color="green", alpha=0.8)
plt.legend()
plt.show()

plt.figure()
plt.title('Final map error: {:.2E}'.format(error_after))
plt.plot(x.flatten(),y_true.flatten(),'*--',label='true data', alpha=0.8)
plt.plot(x.flatten(),y_measured.flatten(),'*--',label='measured data', alpha=0.4)
plt.plot(x.flatten(),map_of_x_after.flatten(),'*--',label='final map output', color="red", alpha=0.8)
plt.legend()
plt.show()