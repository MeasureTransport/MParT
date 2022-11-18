classdef FixedMultiIndexSetTest < matlab.unittest.TestCase
    % Copyright 2014 - 2016 The MathWorks, Inc.

    methods ( Test )

        function size( testCase )
            dim = 2;
            maxOrder = 3;
            mset = FixedMultiIndexSet(dim,maxOrder);
            testCase.verifyEqual( double(mset.Size()), ((maxOrder+1)*(maxOrder+2)/2) );
        end
    end
end
