from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Make target samples
num_points = 5000
z = np.random.randn(2,num_points)
x1 = z[0]
x2 = z[1] + z[0]**2
x = np.vstack([x1,x2])

# For plotting and computing reference density 
rho = multivariate_normal(np.zeros(2),np.eye(2))  #standard normal
t = np.linspace(-5,5,100)
grid = np.meshgrid(t,t)
rho_t = rho.pdf(np.dstack(grid))

# Set-up map and initize map coefficients
opts = MapOptions()

multis_1 = np.array([[0],[1]])  # linear
multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

mset_1 = MultiIndexSet(multis_1).fix(True)
mset_2 = MultiIndexSet(multis_2).fix(True)

map_1 = CreateComponent(mset_1, opts)
map_2 = CreateComponent(mset_2, opts)

tri_map = TriangularMap([map_1,map_2])

#tri_map = CreateTriangular(2,2,2,opts)
coeffs = np.zeros(tri_map.numCoeffs)
tri_map.SetCoeffs(coeffs)


# KL divergence objective
def objective(coeffs, tri_map, x):
    tri_map.SetCoeffs(coeffs)
    map_of_x = tri_map.Evaluate(x)
    rho_of_map_of_x = rho.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)
    return -np.sum(rho_of_map_of_x + log_det)/num_points

# Before optimization
map_of_x = tri_map.Evaluate(x) 
plt.figure()
plt.contour(*grid, rho_t)
plt.scatter(x[0],x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()

# Optimize
print('Starting coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(objective(tri_map.CoeffMap(), tri_map, x)))

res = minimize(objective, tri_map.CoeffMap(), args=(tri_map, x), method="Nelder-Mead")
print('Final coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(objective(tri_map.CoeffMap(), tri_map, x)))

# After optimization
map_of_x = tri_map.Evaluate(x) 
plt.figure()
plt.contour(*grid, rho_t)
plt.scatter(map_of_x[0],map_of_x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()
