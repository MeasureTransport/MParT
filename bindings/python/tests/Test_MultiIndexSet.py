# MultiIndex test

import mpart
import numpy as np
import unittest

class TestStringMethods(unittest.TestCase):

    
    def test_at(self):
        multis = np.array([[0, 1], [2, 0]])
        mset = mpart.MultiIndexSet(multis)
        assert mset.at(0).tolist() == [0,1]
        assert mset.at(1).tolist() == [2,0]
        
        noneLim = mpart.NoneLim()
        mset = mpart.MultiIndexSet.CreateTensorProduct(2,1,noneLim)
        assert mset.at(0).tolist() == [0,0]
        assert mset.at(1).tolist() == [0,1]
        assert mset.at(2).tolist() == [1,0]
        assert mset.at(3).tolist() == [1,1]

        mset = mpart.MultiIndexSet.CreateTotal(2,1,noneLim)
        assert mset.at(0).tolist() == [0,0]
        assert mset.at(1).tolist() == [0,1]
        assert mset.at(2).tolist() == [1,0]


    def test_max_degrees(self):
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