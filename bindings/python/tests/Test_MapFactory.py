

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def test_CreateComponent(self):
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)
        # Set MapOptions and make map
        opts = mpart.MapOptions()
        component = mpart.CreateComponent(mset.fix(True), opts)
        assert component.numCoeffs == 4
        assert np.all(component.CoeffMap() == [0.,0.,0.,0.])




    def test_CreateTriangular(self):
        opts = mpart.MapOptions()

        # multis_1 = np.array([[0],[1]])  # linear
        # multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

        # mset_1 = mpart.MultiIndexSet(multis_1).fix(True)
        # mset_2 = mpart.MultiIndexSet(multis_2).fix(True)

        # map_1 = mpart.CreateComponent(mset_1, opts)
        # map_2 = mpart.CreateComponent(mset_2, opts)

        # triMap = mpart.TriangularMap([map_1,map_2])
        triangular = mpart.CreateTriangular(2,2,2,opts)
        assert triangular.numCoeffs == 2 + 7
        assert triangular.CoeffMap().tolist() == []

        

    

if __name__ == '__main__':
    unittest.main()