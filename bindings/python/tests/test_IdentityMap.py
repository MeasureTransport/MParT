# Test methods of TriangularMap object
import mpart
import numpy as np


inputdim = 20
outputdim = 15
num_samples = 100

x = np.random.randn(inputdim,num_samples)
id_map = mpart.IdentityMap(inputdim,outputdim)

def test_Evaluate():
    y = id_map.Evaluate(x)
    assert y.shape == (outputdim,num_samples)
    assert np.all(y == x[inputdim-outputdim:,:])

def test_LogDeterminant():
    assert id_map.LogDeterminant(x).shape == (num_samples,)
    
def test_Inverse():

    y = id_map.Evaluate(x)
    x_ = id_map.Inverse(np.zeros((1,num_samples)),y)
    assert np.allclose(x_, x[inputdim-outputdim:,:], atol=1E-3)
