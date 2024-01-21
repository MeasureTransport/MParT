# MultiIndex test

import mpart
import numpy as np

multis = np.array([[0, 1], [2, 0]])
msetFromArray = mpart.MultiIndexSet(multis)

noneLim = mpart.NoneLim()
dim = 3
power = 4
msetTensorProduct = mpart.MultiIndexSet.CreateTensorProduct(dim,power,noneLim)
msetTotalOrder = mpart.MultiIndexSet.CreateTotalOrder(dim,power,noneLim)

def test_max_degrees():

    assert np.all(msetFromArray.MaxOrders() == [2,1])
    assert np.all(msetTensorProduct.MaxOrders() == [4,4,4])
    assert np.all(msetTotalOrder.MaxOrders() == [4,4,4])

def test_reduced_margin():
    msetTotalOrder2 = mpart.MultiIndexSet.CreateTotalOrder(dim,power+1,noneLim)
    msetTotalOrder_rm = msetTotalOrder.ReducedMargin()
    assert len(msetTotalOrder_rm) == msetTotalOrder2.Size() - msetTotalOrder.Size()
    assert np.all([midx.sum() == power+1 for midx in msetTotalOrder_rm])
    msetTotalOrder_rm_dim = msetTotalOrder.ReducedMarginDim(2)
    assert np.all([midx.sum() == power+1 for midx in msetTotalOrder_rm_dim])
    assert len(msetTotalOrder_rm_dim) < len(msetTotalOrder_rm)

def test_at():

    assert np.all(msetFromArray[0].tolist() == [0,1])
    assert np.all(msetFromArray[1].tolist() == [2,0])

    assert np.all(msetTensorProduct[0].tolist() == [0,0,0])
    assert np.all(msetTensorProduct[1].tolist() == [0,0,1])
    last_idx = msetTensorProduct.Size()-1
    assert np.all(msetTensorProduct[last_idx].tolist() == [4,4,4])

    assert np.all(msetTotalOrder[0].tolist() == [0,0,0])
    assert np.all(msetTotalOrder[1].tolist() == [0,0,1])
    last_idx = msetTotalOrder.Size()-1
    assert np.all(msetTotalOrder[last_idx].tolist() == [4,0,0])

