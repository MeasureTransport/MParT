from mpart import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Make data
def true_f(x):
    return 2*(x>2).astype('float')

num_points = 100
x = np.linspace(0,4,num_points)

# note: data might not monotone because of the noise, 
# but we assume the true underlying function is monotone.
y = true_f(x) + .4*np.random.randn(num_points)  


# Create multi-index set:
multis = np.array([[0], [1], [2], [3], [4], [5]])
mset = MultiIndexSet(multis)

fixed_mset = mset.fix(True)

# Set MapOptions
opts = MapOptions()

# opts.basisType   = BasisTypes.ProbabilistHermite
# opts.posFuncType = PosFuncTypes.SoftPlus
# opts.quadType    = QuadTypes.AdaptiveSimpson
# opts.quadAbsTol  = 1e-6
# opts.quadRelTol  = 1e-6

map = CreateComponent(fixed_mset, opts)

map_of_x = map.Evaluate(x.reshape(1,num_points))



def objective(coeffs):
    map.SetCoeffs(coeffs)
    map_of_x = map.Evaluate(x.reshape(1,num_points))
    return np.sum((map_of_x - y)**2)/num_points

plt.figure()
plt.plot(x,y,'*--',label='true data')
plt.plot(x,map_of_x.reshape(num_points,),'*--',label='map output')
plt.legend()
plt.title('Starting map error: {:.2E}'.format(objective(map.CoeffMap())))
plt.show()



res = minimize(objective, map.CoeffMap(), method="Nelder-Mead")

map_of_x = map.Evaluate(x.reshape(1,num_points))

plt.figure()
plt.plot(x,y,'*--',label='true data')
plt.plot(x,map_of_x.reshape(num_points,),'*--',label='map output')
plt.legend()
plt.title('Final map error: {:.2E}'.format(objective(map.CoeffMap())))
plt.show()
# def gradient_objective():
#     return 2*map_gradient@(map(x) - y)
# num_coeffs = mset.Size()  # not needed in python example but works

# coeffs = np.array([1.0, 0.5]) 
# # coeffs = None
# map.SetCoeffs(coeffs)

# print('Coeffs set')
# print(map.CoeffMap())
# print('============')

# pts = np.random.randn(2,5)
# print('The points')
# print(pts)
# print('============')

# map_of_pts = map.Evaluate(pts) 
# print('Map of the points')
# print(map_of_pts)
# print('============')

# log_det = map.LogDeterminant(pts)
# print('Log det of map at the points')
# print(log_det)
# print('============')


# # Ways to change coeffs of map
# # coeffs = np.array([2.0, 0.5]) # map does NOT change
# # coeffs[:] = np.array([2.0, 0.5]) # map does change (maybe we don't want it to)
# # coeffs[0] = 2 # map does change (maybe we don't want it to)
# # map.CoeffMap()[:] = np.array([2, 0.5]) # map does change (we want it to)

# coeffs[:] = np.array([4, 0.5])  # changes map, we want this to NOT change the map

# map_of_pts = map.Evaluate(pts)
# print('Map of the points (coeffs changed)')
# print(map_of_pts)
# print('============')



