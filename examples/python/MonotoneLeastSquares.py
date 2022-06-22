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
def objective(coeffs, monotoneMap, x, y_measured, num_points):
    monotoneMap.SetCoeffs(coeffs)
    map_of_x = monotoneMap.Evaluate(x)
    return np.sum((map_of_x - y_measured)**2)/num_points

# Before optimization
map_of_x_before = monotoneMap.Evaluate(x)
error_before = objective(monotoneMap.CoeffMap(), monotoneMap, x, y_measured, num_points)

# Optimize
res = minimize(objective, monotoneMap.CoeffMap(), args=(monotoneMap, x, y_measured, num_points), method="Nelder-Mead")

# After optimization
map_of_x_after = monotoneMap.Evaluate(x)
error_after = objective(monotoneMap.CoeffMap(), monotoneMap, x, y_measured, num_points)

# Plot data
plt.figure()
plt.title('Starting map error: {:.2E} / Final map error: {:.2E}'.format(error_before, error_after))
plt.plot(x.flatten(),y_true.flatten(),'*--',label='true data', alpha=0.8)
plt.plot(x.flatten(),y_measured.flatten(),'*--',label='measured data', alpha=0.4)
plt.plot(x.flatten(),map_of_x_before.flatten(),'*--',label='initial map output', color="green", alpha=0.8)
plt.plot(x.flatten(),map_of_x_after.flatten(),'*--',label='final map output', color="red", alpha=0.8)
plt.legend()
plt.show()
