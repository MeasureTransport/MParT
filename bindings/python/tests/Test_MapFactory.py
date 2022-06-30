

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def test_CreateComponent_basisType(self):
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)

        # Set MapOptions and make map
        opts = mpart.MapOptions()
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

        opts.basisType = mpart.BasisTypes.HermiteFunctions
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])
        
        opts.basisType = mpart.BasisTypes.PhysicistHermite
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    def test_CreateComponent_posFuncType(self):
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)

        # Set MapOptions and make map
        opts = mpart.MapOptions()
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

        opts.posFuncType = mpart.PosFuncTypes.Exp
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    def test_CreateComponent_quadTypes(self):
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)

        # Set MapOptions and make map
        opts = mpart.MapOptions()
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

        opts.quadType = mpart.QuadTypes.AdaptiveClenshawCurtis
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])
        
        opts.quadType = mpart.QuadTypes.ClenshawCurtis
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])


    def test_CreateTriangular(self):
        opts = mpart.MapOptions()
        triangular = mpart.CreateTriangular(2,2,2,opts)
        assert triangular.numCoeffs == 2 + 7
        assert triangular.CoeffMap().tolist() == []

        

    

if __name__ == '__main__':
    unittest.main()