from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Make target samples for training
num_points = 1000
z = np.random.randn(2,num_points)
x1 = z[0]
x2 = z[1] + z[0]**2
x = np.vstack([x1,x2])


# Make target samples for testing
test_z = np.random.randn(2,10000)
test_x1 = test_z[0]
test_x2 = test_z[1] + test_z[0]**2
test_x = np.vstack([test_x1,test_x2])


# For plotting and computing reference density 
rho = multivariate_normal(np.zeros(2),np.eye(2))  #standard normal
t = np.linspace(-5,5,100)
grid = np.meshgrid(t,t)
rho_t = rho.pdf(np.dstack(grid))



# Set-up map and initize map coefficients
opts = MapOptions()
tri_map = CreateTriangular(2,2,2,opts)
coeffs = np.zeros(tri_map.numCoeffs)
tri_map.SetCoeffs(coeffs)


# KL divergence objective
def obj(coeffs, tri_map, x):
    tri_map.SetCoeffs(coeffs)
    map_of_x = tri_map.Evaluate(x)
    rho_of_map_of_x = rho.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)
    return -np.sum(rho_of_map_of_x + log_det)/num_points


def grad_obj(coeffs, tri_map, x):
    tri_map.SetCoeffs(coeffs)
    map_of_x = tri_map.Evaluate(x)
    grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)
    grad_log_det = tri_map.LogDeterminantCoeffGrad(x)
    return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points



# Before optimization
map_of_x = tri_map.Evaluate(x) 
plt.figure()
plt.contour(*grid, rho_t)
plt.scatter(test_x[0],test_x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()

# Optimize
print('Starting coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(obj(tri_map.CoeffMap(), tri_map, x)))
print('==================')

options={'gtol': 1e-16, 'disp': True}
res = minimize(obj, tri_map.CoeffMap(), args=(tri_map, x), jac=grad_obj, method='BFGS', options=options)

print('Final coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(obj(tri_map.CoeffMap(), tri_map, x)))
print('==================')
# After optimization
map_of_test_x = tri_map.Evaluate(test_x)
plt.figure()
plt.contour(*grid, rho_t)
plt.scatter(map_of_test_x[0],map_of_test_x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()


print('==================')

mean_of_map = np.mean(map_of_test_x,1)
print("Mean of normalized test samples")
print(mean_of_map)
print('==================')
print("Cov of normalized test samples")
cov_of_map = np.cov(map_of_test_x)
print(cov_of_map)

