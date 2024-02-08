

import mpart
import numpy as np
import math
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

    # ClenshawCurtis
    opts.quadType = mpart.QuadTypes.ClenshawCurtis
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])

    # AdaptiveClenshawCurtis
    opts.quadType = mpart.QuadTypes.AdaptiveClenshawCurtis
    component = mpart.CreateComponent(mset.fix(True), opts)
    assert component.numCoeffs == 4
    assert np.all(component.CoeffMap() == [0.,0.,0.,0.])


def test_CreateTriangular():
    triangular = mpart.CreateTriangular(2,2,2,opts)
    assert triangular.numCoeffs == 2 + 7 


def test_CreateSingleEntryMap_1():
    dim = 7
    activeInd = 1

    mset_csemap = mpart.MultiIndexSet.CreateTotalOrder(activeInd, 3, noneLim)  
    component = mpart.CreateComponent(mset_csemap.fix(True), opts)

    single_entry_map = mpart.CreateSingleEntryMap(dim, activeInd, component)
    assert single_entry_map.numCoeffs == component.numCoeffs

def test_CreateSingleEntryMap_2():
    dim = 7
    activeInd = 7

    mset_csemap = mpart.MultiIndexSet.CreateTotalOrder(activeInd, 3, noneLim)  
    component = mpart.CreateComponent(mset_csemap.fix(True), opts)

    single_entry_map = mpart.CreateSingleEntryMap(dim, activeInd, component)
    assert single_entry_map.numCoeffs == component.numCoeffs

def test_CreateSingleEntryMap_3():
    dim = 7
    activeInd = 4

    mset_csemap = mpart.MultiIndexSet.CreateTotalOrder(activeInd, 3, noneLim)  
    component = mpart.CreateComponent(mset_csemap.fix(True), opts)

    single_entry_map = mpart.CreateSingleEntryMap(dim, activeInd, component)
    assert single_entry_map.numCoeffs == component.numCoeffs

def test_CreateSigmoidMaps():
    input_dim = 6
    num_sigmoid = 5
    centers_len = 2+num_sigmoid*(num_sigmoid+1)//2
    max_degree = 3
    centers = np.zeros(centers_len)
    center_idx = 0
    bound = 3.
    # Edge terms
    centers[0] = -bound
    centers[1] =  bound
    centers[2] = 0.
    # Sigmoid terms
    for order in range(2,num_sigmoid+1):
        for j in range(order):
            centers[center_idx] = 1.9*bound*(j-(order-1)/2)/(order-1)
            center_idx += 1
    opts = mpart.MapOptions()
    opts.basisType = mpart.BasisTypes.HermiteFunctions
    sig = mpart.CreateSigmoidComponent(input_dim, max_degree, centers, opts)
    expected_num_coeffs = math.comb(input_dim+max_degree, input_dim)
    assert sig.numCoeffs == expected_num_coeffs
    mset_diag = mpart.MultiIndexSet.CreateNonzeroDiagTotalOrder(input_dim, max_degree).fix(True)
    mset_off = mpart.FixedMultiIndexSet(input_dim-1, max_degree)
    sig_mset = mpart.CreateSigmoidComponent(mset_off, mset_diag, centers, opts)
    assert sig_mset.numCoeffs == mset_diag.Size() + mset_off.Size()
    output_dim = input_dim
    centers_total = np.column_stack([centers for _ in range(output_dim)])
    sig_trimap = mpart.CreateSigmoidTriangular(input_dim, output_dim, max_degree, centers_total, opts)
    expected_num_coeffs = np.sum([math.comb(d+max_degree, d) for d in range(1, input_dim+1)])
    assert sig_trimap.numCoeffs == expected_num_coeffs