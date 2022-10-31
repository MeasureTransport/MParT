# Test methods of TriangularMap object
import mpart
import numpy as np



noneLim = mpart.NoneLim()
opts = mpart.MapOptions()
maxDegree = 1
dim = 7
lrcRank = 2
summary_mat = np.random.randn(lrcRank, dim-1)

summary_fun = mpart.AffineFunction(np.asfortranarray(summary_mat))
mset_csemap = mpart.MultiIndexSet.CreateTotalOrder(lrcRank+1, maxDegree, noneLim)

component = mpart.CreateComponent(mset_csemap.fix(True), opts)
summarized_component = mpart.SummarizedMap(summary_fun, component)

num_samples = 100
pts = np.random.randn(dim, num_samples)

coeffs = np.random.randn(summarized_component.numCoeffs)
summarized_component.SetCoeffs(coeffs)

def test_numCoeffs():
    assert summarized_component.numCoeffs == component.numCoeffs


def test_CoeffsMap():
    
    coeffs = np.random.randn(summarized_component.numCoeffs)
    summarized_component.SetCoeffs(coeffs)
    assert np.all(summarized_component.CoeffMap() == coeffs)


def test_Evaluate():

    eval = summarized_component.Evaluate(pts)
    summary = summary_fun.Evaluate(pts[:dim-1,:])
    new_pts = np.vstack([summary,pts[dim-1,:]])
    eval_ = component.Evaluate(new_pts)

    assert eval.shape == (1,num_samples)
    assert np.allclose(eval_, eval, atol=1E-6)


def test_LogDeterminant():

    logdet = summarized_component.LogDeterminant(pts)
    summary = summary_fun.Evaluate(pts[:dim-1,:])
    new_pts = np.vstack([summary,pts[dim-1,:]])
    logdet_ = component.LogDeterminant(new_pts)

    assert logdet.shape == (num_samples,)
    assert np.allclose(logdet_, logdet, atol=1E-6)
    

def test_Inverse():

    y = summarized_component.Evaluate(pts)
    pts_ = summarized_component.Inverse(pts[:dim-1,:], y)
    assert np.allclose(pts_, pts[-1,:], atol=1E-3)
