from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
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
num_points = 1000
z = np.random.randn(num_points)
x = sinharcsinh(z, loc=-1, scale=1, skew=.5, tail=1)
# x = -2 + .5*z  # For Gaussian test case
# For plotting and computing reference density 
rv = norm()
t = np.linspace(-3,3,100)
rho_t = rv.pdf(t)

# Before optimization
num_bins = 50
plt.figure()
plt.hist(x, num_bins, facecolor='blue', alpha=0.5, density=True, label='Target samples')
plt.plot(t,rho_t,label="Reference density")
plt.legend()
plt.show()

# Create multi-index set:
multis = np.array([[0], [1], [2], [3], [4], [5]])
# multis = np.array([[0], [1]])  # For Gaussian test case
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
opts.basisType = BasisTypes.HermiteFunctions
#opts.basisType = BasisTypes.PhysicistHermite
map = CreateComponent(fixed_mset, opts)

# KL divergence objective
def objective(coeffs):
    map.SetCoeffs(coeffs)
    map_of_x = map.Evaluate(x.reshape(1,num_points))
    rho_of_map_of_x = rv.logpdf(map_of_x)
    log_det = map.LogDeterminant(x.reshape(1,num_points))
    return -np.sum(rho_of_map_of_x + log_det)/num_points

# Optimize
print('Starting coeffs')
print(map.CoeffMap())
print('and error: {:.2E}'.format(objective(map.CoeffMap())))
res = minimize(objective, map.CoeffMap(), method="Nelder-Mead")
print('Final coeffs')
print(map.CoeffMap())
print('and error: {:.2E}'.format(objective(map.CoeffMap())))

# After optimization plot
map_of_x = map.Evaluate(x.reshape(1,num_points))
plt.figure()
plt.hist(map_of_x.reshape(num_points,), num_bins, facecolor='blue', alpha=0.5, density=True, label='Normalized samples')
plt.plot(t,rho_t,label="Reference density")
plt.legend()
plt.show()
