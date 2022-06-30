# MultiIndex test

import mpart
import numpy as np

def test_max_degrees():
    multis = np.array([[0, 1], [2, 0]])
    mset = mpart.MultiIndexSet(multis)
    assert np.all(mset.MaxOrders() == [2,1])
    
    noneLim = mpart.NoneLim()
    mset = mpart.MultiIndexSet.CreateTensorProduct(3,4,noneLim)
    assert np.all(mset.MaxOrders() == [4,4,4])

    mset = mpart.MultiIndexSet.CreateTotalOrder(3,4,noneLim)
    assert np.all(mset.MaxOrders() == [4,4,4])
