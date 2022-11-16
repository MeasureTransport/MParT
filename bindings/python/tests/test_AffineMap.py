# MultiIndex test

import mpart
import numpy as np


def test_ShiftOnlyMap():


    b = np.array([1.,2.]).reshape(2,1)
    shift = mpart.AffineMap(b)
    del b  # checking that AffineMap deep-copied b; see issue 275

    b_ = np.array([1.,2.]).reshape(2,1)
    numPts = 10
    pts = np.random.randn(2, numPts)
    
    evals = shift.Evaluate(pts)
    assert np.allclose(evals, pts + b_, atol=1E-8)

    pts_ = shift.Inverse(pts,evals)
    assert np.allclose(pts_, pts, atol=1E-8)

    logdet = shift.LogDeterminant(pts)
    assert np.allclose(logdet, 0, atol=1E-8)



def test_SquareLinearOnlyMap():

    
    A = np.array([[1.,2.],[-3,1.5]])
    
    scale = mpart.AffineMap(np.asfortranarray(A))
    del A # checking that AffineMap deep-copied b; see issue 275

    A_ = np.array([[1.,2.],[-3,1.5]])
    
    numPts = 10
    pts = np.random.randn(2, numPts)


    evals = scale.Evaluate(pts)
    assert np.allclose(evals, A_@pts, atol=1E-8)

    pts_ = scale.Inverse(pts,evals)
    assert np.allclose(pts_, pts, atol=1E-8)

    logdet = scale.LogDeterminant(pts)
    assert np.allclose(logdet, np.log(np.linalg.det(A_)), atol=1E-8)



def test_FullMap():
    A = np.array([[1.,2.],[-3,1.5]])
    b = np.array([1.,2.]).reshape(2,1)
    affine = mpart.AffineMap(np.asfortranarray(A),b)
    
    del A, b  # checking that AffineMap deep-copied A and b; see issue 275
    
    A_ = np.array([[1.,2.],[-3,1.5]])
    b_ = np.array([1.,2.]).reshape(2,1)
    
    numPts = 10
    pts = np.random.randn(2, numPts)
    evals = affine.Evaluate(pts)
    assert np.allclose(evals, A_@pts + b_, atol=1E-8)

    pts_ = affine.Inverse(pts,evals)
    assert np.allclose(pts_, pts, atol=1E-8)

    logdet = affine.LogDeterminant(pts)
    assert np.allclose(logdet, np.log(np.linalg.det(A_)), atol=1E-8)
