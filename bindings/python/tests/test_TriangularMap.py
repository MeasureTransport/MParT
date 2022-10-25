# Test methods of TriangularMap object
import mpart
import numpy as np

opts = mpart.MapOptions()

multis_1 = np.array([[0],[1]])  # linear
multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

mset_1 = mpart.MultiIndexSet(multis_1).fix(True)
mset_2 = mpart.MultiIndexSet(multis_2).fix(True)

map_1 = mpart.CreateComponent(mset_1, opts)
map_2 = mpart.CreateComponent(mset_2, opts)

triangular = mpart.TriangularMap([map_1,map_2])
num_samples = 100
x = np.random.randn(2,num_samples)


def test_numCoeffs():
    assert triangular.numCoeffs == 2 + 3


def test_CoeffsMap():
    coeffs = np.random.randn(triangular.numCoeffs)
    triangular.SetCoeffs(coeffs)
    assert np.all(triangular.CoeffMap() == coeffs)


def test_Evaluate():
    assert triangular.Evaluate(x).shape == (2,num_samples)


def test_LogDeterminant():
    assert triangular.LogDeterminant(x).shape == (num_samples,)
    
def test_Inverse():
    coeffs = np.random.randn(triangular.numCoeffs)
    triangular.SetCoeffs(coeffs)
    y = triangular.Evaluate(x)
    x_ = triangular.Inverse(np.zeros((0,num_samples)),y)
    assert np.allclose(x_, x, atol=1E-3)
