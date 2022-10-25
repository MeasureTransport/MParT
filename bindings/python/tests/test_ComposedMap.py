# Test methods of ComposedMap object
import mpart
import numpy as np
print("made map")
opts = mpart.MapOptions()

num_maps = 8
dim = 2
maxOrder = 1

maps = []
for i in range(num_maps):
    maps.append(mpart.CreateTriangular(dim,dim,maxOrder,opts))


composed_map = mpart.ComposedMap(maps, num_maps)
composed_map.SetCoeffs(np.random.randn(composed_map.numCoeffs))

num_samples = 100
x = np.random.randn(dim,num_samples)


# def test_numCoeffs():
#     assert triangular.numCoeffs == 2 + 3


# def test_CoeffsMap():
    
#     assert triangular.CoeffMap().tolist() == [0,0,0,0,0]


def test_Evaluate():
    assert composed_map.Evaluate(x).shape == (dim, num_samples)


def test_LogDeterminant():
    assert composed_map.LogDeterminant(x).shape == (num_samples,)
    
def test_Inverse():
    coeffs = np.random.randn(composed_map.numCoeffs)
    composed_map.SetCoeffs(coeffs)
    y = composed_map.Evaluate(x)
    x_ = composed_map.Inverse(np.zeros((0,num_samples)),y)
    assert np.allclose(x_, x, atol=1E-3)
