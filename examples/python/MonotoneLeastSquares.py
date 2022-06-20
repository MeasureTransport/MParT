from mpart import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Make data
def true_f(x):
    return 2*(x>2).astype('float')

num_points = 1000
x = np.linspace(0,4,num_points)

# note: data might not monotone because of the noise, 
# but we assume the true underlying function is monotone.
y = true_f(x) + .4*np.random.randn(num_points)  

# Create multi-index set:
multis = np.array([[0], [1], [2], [3], [4], [5]])
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
map = CreateComponent(fixed_mset, opts)

# Least squares objective
def objective(coeffs):
    map.SetCoeffs(coeffs)
    map_of_x = map.Evaluate(x.reshape(1,num_points))
    return np.sum((map_of_x - y)**2)/num_points

# Before optimization
map_of_x = map.Evaluate(x.reshape(1,num_points))
plt.figure()
plt.plot(x,y,'*--',label='true data')
plt.plot(x,map_of_x.reshape(num_points,),'*--',label='map output')
plt.legend()
plt.title('Starting map error: {:.2E}'.format(objective(map.CoeffMap())))
plt.show()

# Optimize
res = minimize(objective, map.CoeffMap(), method="Nelder-Mead")

# After optimization plot
map_of_x = map.Evaluate(x.reshape(1,num_points))

plt.figure()
plt.plot(x,y,'*--',label='true data')
plt.plot(x,map_of_x.reshape(num_points,),'*--',label='map output')
plt.legend()
plt.title('Final map error: {:.2E}'.format(objective(map.CoeffMap())))
plt.show()

