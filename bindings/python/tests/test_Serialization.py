# Serialization test

import mpart as mt
import numpy as np
import os
import pytest

opts = mt.MapOptions()

multis_1 = np.array([[0],[1]])  # linear
multis_2 = np.array([[0,0],[0,1],[2,0]])  # quadratic in x_1, linear in x_2, matches form of target

mset_1 = mt.MultiIndexSet(multis_1).fix(True)
mset_2 = mt.MultiIndexSet(multis_2).fix(True)

map_1 = mt.CreateComponent(mset_1, opts)
map_2 = mt.CreateComponent(mset_2, opts)

triangular = mt.TriangularMap([map_1,map_2])
triangular.SetCoeffs(np.concatenate([map_1.CoeffMap(), map_2.CoeffMap()]))

@pytest.fixture(autouse=True)
def run_around_tests():
    os.mkdir("_test_serialization")
    yield
    os.remove("_test_serialization/opts.mpart")
    os.remove("_test_serialization/mset_1.mpart")
    os.remove("_test_serialization/mset_2.mpart")
    os.remove("_test_serialization/map_1.mpart")
    os.remove("_test_serialization/map_2.mpart")
    os.remove("_test_serialization/triangular.mpart")
    os.rmdir("_test_serialization")


def test_serialization():
    mt.SerializeMapOptions(opts, "_test_serialization/opts.mpart")
    mt.SerializeFixedMultiIndexSet(mset_1, "_test_serialization/mset_1.mpart")
    mt.SerializeFixedMultiIndexSet(mset_2, "_test_serialization/mset_2.mpart")
    mt.SerializeMapCoeffs(map_1, "_test_serialization/map_1.mpart")
    mt.SerializeMapCoeffs(map_2, "_test_serialization/map_2.mpart")
    mt.SerializeMapCoeffs(triangular, "_test_serialization/triangular.mpart")

    opts_s = mt.DeserializeMapOptions("_test_serialization/opts.mpart")
    mset_1_s = mt.DeserializeFixedMultiIndexSet("_test_serialization/mset_1.mpart")
    mset_2_s = mt.DeserializeFixedMultiIndexSet("_test_serialization/mset_2.mpart")
    map_1_s = mt.DeserializeMapCoeffs("_test_serialization/map_1.mpart")
    map_2_s = mt.DeserializeMapCoeffs("_test_serialization/map_2.mpart")
    triangularCoeff_s = mt.DeserializeMapCoeffs("_test_serialization/triangular.mpart")
    assert (mset_1_s.MaxDegrees() == mset_1.MaxDegrees()).all()
    assert (mset_2_s.MaxDegrees() == mset_2.MaxDegrees()).all()
    assert opts_s.basisType == opts.basisType
    assert opts_s.basisLB == opts.basisLB
    assert opts_s.basisUB == opts.basisUB
    assert opts_s.basisNorm == opts.basisNorm
    assert opts_s.posFuncType == opts.posFuncType
    assert opts_s.quadType == opts.quadType
    assert opts_s.quadAbsTol == opts.quadAbsTol
    assert opts_s.quadRelTol == opts.quadRelTol
    assert opts_s.quadMaxSub == opts.quadMaxSub
    assert opts_s.quadMinSub == opts.quadMinSub
    assert opts_s.quadPts == opts.quadPts
    assert opts_s.contDeriv == opts.contDeriv
    assert (map_1_s == map_1.CoeffMap()).all()
    assert (map_2_s == map_2.CoeffMap()).all()
    assert (triangularCoeff_s == triangular.CoeffMap()).all()