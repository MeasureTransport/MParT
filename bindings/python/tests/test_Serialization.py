# Serialization test

import mpart as mt
import numpy as np
import os
import pytest

import pickle

# Initialize folder name for test
foldername = "_test_serialization/"

# Create MapOptions
options = mt.MapOptions()
options.basisType = mt.BasisTypes.ProbabilistHermite
options.basisNorm = False;

# Create FixedMultiIndexSet and map component
multis = np.array([[0],[1]])
mset= mt.MultiIndexSet(multis)
fixed_mset = mset.fix(True)

component = mt.CreateComponent(fixed_mset, options)
coeffs = component.CoeffMap()
for i in range(len(coeffs)):
    coeffs[i] = 0.5*(i+1)

# Serialize objects

def DeserializeComponent():
    # Deserialize the FixedMultiIndexSet
    # Note that we need to construct the object before calling Deserialize
    fixed_mset = mt.MultiIndexSet(np.array([[0]])).fix(True)
    fixed_mset.Deserialize("fmset.mt")

    # Deserialize the MapOptions
    options = MapOptions()
    options.Deserialize("opts.mt")

    # Deserialize the Map Coefficients and construct the component
    inputDim, outputDim, coeffs = mt.DeserializeMap("comp.mt")
    component = mt.CreateComponent(fixed_mset, options)
    component.SetCoeffs(coeffs)

@pytest.fixture(autouse=True)
def run_around_tests():
    os.mkdir(foldername)
    yield
    for filename in ["opts.mt", "fmset.mt", "comp.mt"]:
        if os.path.isfile(foldername + filename):
            os.remove(foldername + filename)
    os.rmdir(foldername)


def test_serialization():
    fixed_mset.Serialize(foldername + "fmset.mt")
    options.Serialize(foldername + "opts.mt")
    component.Serialize(foldername + "comp.mt")

    options_s = mt.MapOptions()
    fixed_mset_s = mt.MultiIndexSet(np.array([[0]])).fix(True)

    options_s.Deserialize(foldername + "opts.mt")
    fixed_mset_s.Deserialize(foldername + "fmset.mt")
    inputDim_s, outputDim_s, coeffs_s = mt.DeserializeMap(foldername + "comp.mt")
    component_s = mt.CreateComponent(fixed_mset_s, options_s)
    component_s.SetCoeffs(coeffs_s)

    assert (fixed_mset_s.MaxDegrees() == fixed_mset.MaxDegrees()).all()
    assert options_s.basisType == options.basisType
    assert options_s.basisLB == options.basisLB
    assert options_s.basisUB == options.basisUB
    assert options_s.basisNorm == options.basisNorm
    assert options_s.posFuncType == options.posFuncType
    assert options_s.quadType == options.quadType
    assert options_s.quadAbsTol == options.quadAbsTol
    assert options_s.quadRelTol == options.quadRelTol
    assert options_s.quadMaxSub == options.quadMaxSub
    assert options_s.quadMinSub == options.quadMinSub
    assert options_s.quadPts == options.quadPts
    assert options_s.contDeriv == options.contDeriv
    assert component_s.inputDim == inputDim_s
    assert component_s.outputDim == outputDim_s
    assert (component_s.CoeffMap() == coeffs).all()

def test_trimap_saveload():
    opts = mt.MapOptions()

    multis_1 = np.array([[0],[1]])  # linear
    multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target 

    mset_1 = mt.MultiIndexSet(multis_1).fix(True)
    mset_2 = mt.MultiIndexSet(multis_2).fix(True)

    map_1 = mt.CreateComponent(mset_1, opts)
    map_2 = mt.CreateComponent(mset_2, opts)

    tmap = mt.TriangularMap([map_1,map_2],True)
    tmap.SetCoeffs(np.random.randn(tmap.numCoeffs))
    
    # Dump the tmap to a byte string
    tmap_bytes = tmap.ToBytes()

    # Create a copy by reading the byte string
    tmap2 = mt.TriangularMap.FromBytes(tmap_bytes)
    assert np.all(tmap2.CoeffMap() == tmap.CoeffMap())

    # Now try it with pickle
    tmap_bytes = pickle.dumps(tmap)
    tmap2 = pickle.loads(tmap_bytes)
    assert np.all(tmap2.CoeffMap() == tmap.CoeffMap())
