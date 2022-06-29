from mpart import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Make target samples
num_points = 5000
mu = 2
sigma = .5
x = np.random.randn(num_points)

# for plotting
rv = norm(loc=mu,scale=sigma)
t = np.linspace(-3,6,100)
rho_t = rv.pdf(t)

num_bins = 50
# Before optimization plot
plt.figure()
plt.hist(x, num_bins, facecolor='blue', alpha=0.5, density=True, label='Reference samples')
plt.plot(t,rho_t,label="Target density")
plt.legend()
plt.show()

# Create multi-index set
multis = np.array([[0], [1]])  # affine transform enough to capture Gaussian target
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
def objective(coeffs):
    monotoneMap.SetCoeffs(coeffs)
    map_of_x = monotoneMap.Evaluate(x.reshape(1,num_points))
    pi_of_map_of_x = rv.logpdf(map_of_x)
    log_det = monotoneMap.LogDeterminant(x.reshape(1,num_points))
    return -np.sum(pi_of_map_of_x + log_det)/num_points


# Optimize
print('Starting coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap())))
res = minimize(objective, monotoneMap.CoeffMap(), method="Nelder-Mead")
print('Final coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap())))

# After optimization plot
map_of_x = monotoneMap.Evaluate(x.reshape(1,num_points))
plt.figure()
plt.hist(map_of_x.reshape(num_points,), num_bins, facecolor='blue', alpha=0.5, density=True, label='Mapped samples')
plt.plot(t,rho_t,label="Target density")
plt.legend()
plt.show()
