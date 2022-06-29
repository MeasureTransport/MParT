

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    def test_CreateComponent(self):
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTotalOrder(1,3, noneLim)
        # Set MapOptions and make map
        opts = mpart.MapOptions()
        monotoneMap = mpart.CreateComponent(mset.fix(True), opts)
        assert monotoneMap.numCoeffs == 4
        assert np.all(monotoneMap.CoeffMap() == [0.,0.,0.,0.])




    def test_CreateTriangular(self):
        multis = np.array([[0, 1], [2, 0]])
        mset = mpart.MultiIndexSet(multis)
        assert np.all(mset.MaxOrders() == [2,1])
        
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTensorProduct(3,4,noneLim)
        assert np.all(mset.MaxOrders() == [4,4,4])

        mset = mpart.MultiIndexSet.CreateTotal(3,4,noneLim)
        assert np.all(mset.MaxOrders() == [4,4,4])

    

if __name__ == '__main__':
    unittest.main()