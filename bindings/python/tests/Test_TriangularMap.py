

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def setUp(self):
    
        opts = mpart.MapOptions()

        multis_1 = np.array([[0],[1]])  # linear
        multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

        mset_1 = mpart.MultiIndexSet(multis_1).fix(True)
        mset_2 = mpart.MultiIndexSet(multis_2).fix(True)

        map_1 = mpart.CreateComponent(mset_1, opts)
        map_2 = mpart.CreateComponent(mset_2, opts)

        self.triangular = mpart.TriangularMap([map_1,map_2])
        self.num_samples = 100
        self.x = np.random.randn(2,self.num_samples)


    def test_numCoeffs(self):
        assert self.triangular.numCoeffs == 2 + 3


    def test_CoeffsMap(self):
        self.triangular.SetCoeffs(np.zeros(self.triangular.numCoeffs))
        assert self.triangular.CoeffMap().tolist() == [0,0,0,0,0]


    def test_Evaluate(self):
        assert self.triangular.Evaluate(self.x).shape == (2,self.num_samples)


    def test_LogDeterminant(self):
        assert self.triangular.LogDeterminant(self.x).shape == (self.num_samples,)
        
    def test_Inverse(self):
        coeffs = np.random.randn(self.triangular.numCoeffs)
        self.triangular.SetCoeffs(coeffs)
        y = self.triangular.Evaluate(self.x)
        x_ = self.triangular.Inverse(np.zeros((0,self.num_samples)),y)
        assert np.allclose(x_, self.x, atol=1E-3)

        



        

    

if __name__ == '__main__':
    unittest.main()