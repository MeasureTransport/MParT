# Test methods of ComposedMap object
import mpart
import numpy as np
print("made map")
opts = mpart.MapOptions()

num_maps = 4
dim = 2
maxOrder = 1

maps = []
coeffs_ = []
numCoeffs = 0
for i in range(num_maps):
    map = mpart.CreateTriangular(dim,dim,maxOrder,opts)
    coeffs = np.random.randn(map.numCoeffs)
    map.SetCoeffs(coeffs)
    maps.append(map)
    coeffs_.append(coeffs)
    numCoeffs += map.numCoeffs


moveCoeffs = True
composed_map = mpart.ComposedMap(maps, moveCoeffs, num_maps)
all_coeffs = np.hstack(coeffs_).flatten()

num_samples = 100
x = np.random.randn(dim, num_samples)

def test_numCoeffs():
    assert composed_map.numCoeffs == numCoeffs

def test_CoeffsMap():
    assert np.all(composed_map.CoeffMap() == all_coeffs)


def test_Evaluate():
    assert composed_map.Evaluate(x).shape == (dim, num_samples)
    x_ = x.copy()
    for map_ in maps:
        x_ = map_.Evaluate(x_)
    assert np.all(composed_map.Evaluate(x) == x_)


def test_LogDeterminant():
    assert composed_map.LogDeterminant(x).shape == (num_samples,)

def test_Inverse():

    y = composed_map.Evaluate(x)
    x_ = composed_map.Inverse(np.zeros((1,num_samples)),y)
    assert np.allclose(x_, x, atol=1E-3)
