

import mpart
import numpy as np
import unittest

noneLim = mpart.NoneLim()
mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)
opts = mpart.MapOptions()


def test_CreateComponent_basisType():

    # HermiteFunctions
    opts.basisType = mpart.BasisTypes.HermiteFunctions
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])
    
    # PhysicistHermite
    opts.basisType = mpart.BasisTypes.PhysicistHermite
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    # ProbabilistHermite
    opts.basisType = mpart.BasisTypes.ProbabilistHermite
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])


def test_CreateComponent_posFuncType():

    # Exp
    opts.posFuncType = mpart.PosFuncTypes.Exp
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    # SoftPlus 
    opts.posFuncType = mpart.PosFuncTypes.SoftPlus
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

def test_CreateComponent_quadTypes():

    # AdaptiveSimpson
    opts.quadType = mpart.QuadTypes.AdaptiveSimpson
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    # # ClenshawCurtis
    # opts.quadType = mpart.QuadTypes.ClenshawCurtis
    # component = mpart.CreateComponent(mset.fix(True), opts)
    # assert component.numCoeffs == 4
    # assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    # # AdaptiveClenshawCurtis
    # opts.quadType = mpart.QuadTypes.AdaptiveClenshawCurtis
    # component = mpart.CreateComponent(mset.fix(True), opts)
    # assert component.numCoeffs == 4
    # assert np.all(component.CoeffMap() == [0.,0.,0.,0.])



def test_CreateTriangular():
    triangular = mpart.CreateTriangular(2,2,2,opts)
    assert triangular.numCoeffs == 2 + 7 

