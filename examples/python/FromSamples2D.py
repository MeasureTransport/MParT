from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# sinh-arcsinh function
def sinharcsinh(z,loc,scale,skew,tail):
    '''
    To make skewed and/or non-Gaussian tailed test distributions
    skew \in R, skew > 0 leads to positive (right tilted) skew, skew < 0 leads to negative (left tilted) skew
    tail > 0, tail < 1 leads to light tails, tail > 1 leads to heavy tails.
    skew = 0, tail = 1 leads to affine function x = loc + scale*z
    See for more info: Jones, M. Chris, and Arthur Pewsey. "Sinh-arcsinh distributions." Biometrika 96.4 (2009): 761-780.
    '''
    f0 = np.sinh(tail*np.arcsinh(2))
    f = (2/f0)*np.sinh(tail*(np.arcsinh(z) + skew))
    x = loc + scale*f
    return x


# Make target samples
num_points = 5000
z = np.random.randn(2,num_points)
x1 = z[0]
x2 = z[1] + z[0]**2
x = np.vstack([x1,x2])
# x = -2 + .5*z  # For Gaussian test case
# For plotting and computing reference density 
rv = multivariate_normal(np.zeros(2),np.eye(2))
t = np.linspace(-3,3,100)
T1, T2 = np.meshgrid(t,t)
rho_t = rv.pdf(np.dstack((T1,T2)))

# Before optimization
num_bins = 10
plt.figure()
#plt.hist2d(x[0],x[1], num_bins, facecolor='blue', alpha=0.5, density=True, label='Target samples')
plt.contour(T1,T2,rho_t,label="Reference density")
plt.scatter(x[0],x[1], facecolor='blue', alpha=0.1, label='Target samples')
# plt.legend()
plt.show()


# Set MapOptions and make map
opts = MapOptions()
multis_1 = np.array([[0],[1]])
multis_2 = np.array([[0,0],[0,1],[2,0]])

mset_1 = MultiIndexSet(multis_1).fix(True)
mset_2 = MultiIndexSet(multis_2).fix(True)

map_1 = CreateComponent(mset_1,opts)
map_2 = CreateComponent(mset_2,opts)

tri_map = TriangularMap([map_1,map_2])
# tri_map = CreateTriangular(2,2,2,opts)
coeffs = np.zeros(tri_map.numCoeffs)
tri_map.SetCoeffs(coeffs)


# KL divergence objective
def objective(coeffs):
    tri_map.SetCoeffs(coeffs)
    map_of_x = tri_map.Evaluate(x)
    rho_of_map_of_x = rv.logpdf(map_of_x.T)
    log_det = tri_map.LogDeterminant(x)
    return -np.sum(rho_of_map_of_x + log_det)/num_points

objective(coeffs)

# Optimize
print('Starting coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(objective(tri_map.CoeffMap())))
res = minimize(objective, tri_map.CoeffMap(), method="Nelder-Mead")
print('Final coeffs')
print(tri_map.CoeffMap())
print('and error: {:.2E}'.format(objective(tri_map.CoeffMap())))

# Before optimization
map_of_x = tri_map.Evaluate(x) 
plt.figure()
#plt.hist2d(x[0],x[1], num_bins, facecolor='blue', alpha=0.5, density=True, label='Target samples')
plt.contour(T1,T2,rho_t,label="Reference density")
plt.scatter(map_of_x[0],map_of_x[1], facecolor='blue', alpha=0.1, label='Target samples')
plt.legend()
plt.show()
