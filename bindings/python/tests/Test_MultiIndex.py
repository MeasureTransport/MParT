# MultiIndex test

import mpart
import unittest

class TestStringMethods(unittest.TestCase):

    def test_sum(self):
        
        idx = mpart.MultiIndex(30,2)

        
        assert idx.sum() == 60

    def test_max(self):

        idx = mpart.MultiIndex(30,2)
        idx[0] = 100

        assert idx.max() == 100

    def test_count_nonzero(self):
        idx = mpart.MultiIndex(30,0)

        idx[3] = 1
        idx[26] = 3
        assert idx.count_nonzero() == 2



