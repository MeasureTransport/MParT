# FixedMultiIndexSet test

import mpart
import numpy as np
import unittest

class TestFixedMsetMethods(unittest.TestCase):

    def test_max_degrees(self):
        multis = np.array([[0, 1], [2, 0]])
        mset = mpart.MultiIndexSet(multis)
        fixed_mset = mset.fix(True)
        assert np.all(fixed_mset == [1,2])

if __name__ == '__main__':
    unittest.main()