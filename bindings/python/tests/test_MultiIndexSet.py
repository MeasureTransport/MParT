# MultiIndex test

import mpart
import numpy as np

multis = np.array([[0, 1], [2, 0]])
msetFromArray = mpart.MultiIndexSet(multis)

noneLim = mpart.NoneLim()
msetTensorProduct = mpart.MultiIndexSet.CreateTensorProduct(3,4,noneLim)
msetTotalOrder = mpart.MultiIndexSet.CreateTotalOrder(3,4,noneLim)

def test_max_degrees():

    assert np.all(msetFromArray.MaxOrders() == [2,1])
    assert np.all(msetTensorProduct.MaxOrders() == [4,4,4])
    assert np.all(msetTotalOrder.MaxOrders() == [4,4,4])


def test_at():

    assert np.all(msetFromArray.at(0).tolist() == [0,1])
    assert np.all(msetFromArray.at(1).tolist() == [2,0])

    assert np.all(msetTensorProduct.at(0).tolist() == [0,0,0])
    assert np.all(msetTensorProduct.at(1).tolist() == [0,0,1])
    last_idx = msetTensorProduct.Size()-1
    assert np.all(msetTensorProduct.at(last_idx).tolist() == [4,4,4])

    assert np.all(msetTotalOrder.at(0).tolist() == [0,0,0])
    assert np.all(msetTotalOrder.at(1).tolist() == [0,0,1])
    last_idx = msetTotalOrder.Size()-1
    assert np.all(msetTotalOrder.at(last_idx).tolist() == [4,0,0])

