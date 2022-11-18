# MultiIndex test

import mpart
import numpy as np


def test_ShiftOnlyFunction():


    b = np.array([1.,2.]).reshape(2,1)
    shift = mpart.AffineFunction(b)
    del b  # checking that AffineFunction deep-copied b; see issue 275

    b_ = np.array([1.,2.]).reshape(2,1)
    numPts = 10
    pts = np.random.randn(2, numPts)
    evals = shift.Evaluate(pts)

    assert np.allclose(evals, pts + b_, atol=1E-8)



def test_SquareLinearOnlyFunction():

    
    A = np.array([[1.,2.],[-3,1.5]])
    
    scale = mpart.AffineFunction(np.asfortranarray(A))
    del A # checking that AffineFunction deep-copied b; see issue 275

    A_ = np.array([[1.,2.],[-3,1.5]])
    
    numPts = 10
    pts = np.random.randn(2, numPts)
    evals = scale.Evaluate(pts)

    assert np.allclose(evals, A_@pts, atol=1E-8)


def test_RectLinearOnlyFunction():
    A = np.array([[1.,1.,3],[-3,1.5,-3.4]])
    
    scale = mpart.AffineFunction(np.asfortranarray(A))
    del A # checking that AffineFunction deep-copied b; see issue 275

    A_ = np.array([[1.,1.,3],[-3,1.5,-3.4]])
    
    numPts = 10
    pts = np.random.randn(3, numPts)
    evals = scale.Evaluate(pts)

    assert np.allclose(evals, A_@pts, atol=1E-8)


def test_FullFunction():
    A = np.array([[1.,2.],[-3,1.5]])
    b = np.array([1.,2.]).reshape(2,1)
    affine = mpart.AffineFunction(np.asfortranarray(A),b)
    
    del A, b  # checking that AffineFunction deep-copied A and b; see issue 275
    
    A_ = np.array([[1.,2.],[-3,1.5]])
    b_ = np.array([1.,2.]).reshape(2,1)
    
    numPts = 10
    pts = np.random.randn(2, numPts)
    evals = affine.Evaluate(pts)

    assert np.allclose(evals, A_@pts + b_, atol=1E-8)


# recreating bug example from issue 275
def test_Issue275():
    
    dim = 100
    numPts = 10
    # lists to hold matrices and functions
    mats_ = []
    funs = []
    for idx in range(2, dim):

        # create summary_matrix
        A = np.random.randn(idx, idx)
        Q, _ = np.linalg.qr(A)
        mat = Q[:2,:].copy()
        mat_ = Q[:2,:].copy()
        mats_.append(mat_)  

        # create summary function
        fun = mpart.AffineFunction(np.asfortranarray(mat))
        funs.append(fun)

    for mat_,fun in zip(mats_, funs):

        pts = np.random.randn(mat_.shape[1], numPts)
        evals = fun.Evaluate(pts)

        assert np.allclose(evals, mat_@pts, atol=1E-8)
