

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    # def test_CreateComponent(self):
    #     noneLim = mpart.NoneLim()
    #     mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)
    #     # Set MapOptions and make map
    #     opts = mpart.MapOptions()
    #     component = mpart.CreateComponent(mset.fix(True), opts)
    #     assert component.numCoeffs == 4
    #     assert np.all(component.CoeffMap() == [0.,0.,0.,0.])




    def test_init(self):
        opts = mpart.MapOptions()

        multis_1 = np.array([[0],[1]])  # linear
        multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

        mset_1 = mpart.MultiIndexSet(multis_1).fix(True)
        mset_2 = mpart.MultiIndexSet(multis_2).fix(True)

        map_1 = mpart.CreateComponent(mset_1, opts)
        map_2 = mpart.CreateComponent(mset_2, opts)

        triangular = mpart.TriangularMap([map_1,map_2])
        
        assert triangular.numCoeffs == 2 + 3
        assert triangular.CoeffMap().tolist() == []

        triangular.SetCoeffs(np.zeros(triangular.numCoeffs))
        assert triangular.CoeffMap().tolist() == [0,0,0,0,0]

        x = np.random.randn(2,100)
        np.all(np.abs(triangular.Evaluate(x) - x) < 1E-12)

        triangular.SetCoeffs(np.array([0.,1.,0.,0.,0.]))
        y = triangular.Evaluate(x)
        x_ = triangular.InverseInplace(y)


        

    

if __name__ == '__main__':
    unittest.main()