# MultiIndex test

import mpart
import unittest

class TestStringMethods(unittest.TestCase):

    # This one passes :-)
    def test_sum_1(self):
        
        idx = mpart.MultiIndex(30,2)
        idx[0] = 2  # should do nothing
        assert idx.sum() == 60

    # This one fails :-(
    def test_sum_2(self):
        
        idx = mpart.MultiIndex(30,2)

        assert idx.sum() == 60

    # This one passes :-)
    def test_max_1(self):

        idx = mpart.MultiIndex(30,2)
        idx[0] = 100

        assert idx.max() == 100
    
    def test_max_2(self):

        idx = mpart.MultiIndex(30,2)
        

        assert idx.max() == 2

    def test_count_nonzero(self):
        idx = mpart.MultiIndex(30,0)

        idx[3] = 1
        idx[26] = 3
        assert idx.count_nonzero() == 2



if __name__ == '__main__':
    unittest.main()