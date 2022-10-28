# Test methods of TriangularMap object
import mpart
import numpy as np



noneLim = mpart.NoneLim()
opts = mpart.MapOptions()
maxDegree = 2
dim = 7
lrcRank = 2
summary_mat = np.random.randn(lrcRank, dim-1)

summary_fun = mpart.AffineFunction(np.asfortranarray(summary_mat))
mset_csemap = mpart.MultiIndexSet.CreateTotalOrder(lrcRank+1, maxDegree, noneLim)

component = mpart.CreateComponent(mset_csemap.fix(True), opts)
summarized_component = mpart.SummarizedMap(summary_fun, component)

num_samples = 100
x = np.random.randn(dim, num_samples)
print(summarized_component.numCoeffs)
summarized_component.SetCoeffs(np.zeros(summarized_component.numCoeffs))

def test_numCoeffs():
    assert summarized_component.numCoeffs == 2 + 3


def test_CoeffsMap():
    summarized_component.SetCoeffs(np.zeros(summarized_component.numCoeffs))
    assert summarized_component.CoeffMap().tolist() == np.zeros(summarized_component.numCoeffs).tolist()


def test_Evaluate():
    assert summarized_component.Evaluate(x).shape == (1,num_samples)


def test_LogDeterminant():
    assert summarized_component.LogDeterminant(x).shape == (num_samples,)
    

def test_Inverse():
    coeffs = np.random.randn(summarized_component.numCoeffs)
    summarized_component.SetCoeffs(coeffs)
    y = summarized_component.Evaluate(x)
    x_ = summarized_component.Inverse(np.zeros((0,num_samples)),y)
    assert np.allclose(x_, x, atol=1E-3)
