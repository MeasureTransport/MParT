from mpart import *
import numpy as np
import matplotlib.pyplot as plt

# Create multi-index set:
multis = np.array([[0, 1], [2, 0]])
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
# num_coeffs = mset.Size()  # not needed in python example but works

coeffs = np.array([1.0, 0.5]) 
# coeffs = None
map.SetCoeffs(coeffs)

print('Coeffs set')
print(map.CoeffMap())
print('============')

pts = np.random.randn(2,5)
print('The points')
print(pts)
print('============')

map_of_pts = map.Evaluate(pts) 
print('Map of the points')
print(map_of_pts)
print('============')

log_det = map.LogDeterminant(pts)
print('Log det of map at the points')
print(log_det)
print('============')


# Ways to change coeffs of map
# coeffs = np.array([2.0, 0.5]) # map does NOT change
# coeffs[:] = np.array([2.0, 0.5]) # map does change (maybe we don't want it to)
# coeffs[0] = 2 # map does change (maybe we don't want it to)
# map.CoeffMap()[:] = np.array([2, 0.5]) # map does change (we want it to)

coeffs[:] = np.array([4, 0.5])  # changes map, we want this to NOT change the map

map_of_pts = map.Evaluate(pts)
print('Map of the points (coeffs changed)')
print(map_of_pts)
print('============')



