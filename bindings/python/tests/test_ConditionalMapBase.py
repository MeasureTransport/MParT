# Test methods of ConditionalMapBase object
import mpart
import numpy as np

opts = mpart.MapOptions()

multis = np.array([[0],[1]])  # linear
mset = mpart.MultiIndexSet(multis).fix(True)

component = mpart.CreateComponent(mset, opts)
num_samples = 100
x = np.random.randn(1,num_samples)


def test_numCoeffs():
    assert component.numCoeffs == 2


def test_CoeffsMap():
    component.SetCoeffs(np.zeros(component.numCoeffs))
    assert component.CoeffMap().tolist() == [0,0]

    coeffs = np.random.randn(component.numCoeffs)
    component.SetCoeffs(coeffs)
    assert np.all(component.CoeffMap() == coeffs)


def test_Evaluate():
    assert component.Evaluate(x).shape == (1,num_samples)


def test_LogDeterminant():
    assert component.LogDeterminant(x).shape == (num_samples,)
    

def test_Inverse():
    coeffs = np.random.randn(component.numCoeffs)
    component.SetCoeffs(coeffs)
    y = component.Evaluate(x)
    x_ = component.Inverse(np.zeros((0,num_samples)),y)
    assert np.allclose(x_, x, atol=1E-3)

