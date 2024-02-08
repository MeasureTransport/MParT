classdef MultiIndexSetTest < matlab.unittest.TestCase
    % Copyright 2014 - 2016 The MathWorks, Inc.

    methods ( Test )

        function TestCreateTotalOrder( testCase )
            dim = 4;
            totalOrder = 5;
            % Test different CreateTotalOrder methods
            mset1 = MultiIndexSet.CreateTotalOrder( dim, totalOrder );
            expected_len_TO = uint32(nchoosek( dim + totalOrder, totalOrder ));
            testCase.verifyEqual( mset1.Size(), expected_len_TO );
            mset2 = MultiIndexSet.CreateNonzeroDiagTotalOrder( dim, totalOrder );
            expected_len_NzDTO = uint32(nchoosek( dim + totalOrder, totalOrder ) - nchoosek(dim + totalOrder - 1, totalOrder));
            testCase.verifyEqual( mset2.Size(), expected_len_NzDTO );
            mset3 = MultiIndexSet.CreateSeparableTotalOrder( dim, totalOrder );
            expected_len_STO = uint32(nchoosek( dim + totalOrder - 1, totalOrder ) + totalOrder);
            testCase.verifyEqual( mset3.Size(), expected_len_STO );
        end

        function TestDiagonalIndices( testCase )
            dim = 4; totalOrder = 5;
            mset1 = MultiIndexSet.CreateTotalOrder( dim, totalOrder );
            mset2 = MultiIndexSet.CreateNonzeroDiagTotalOrder( dim, totalOrder );
            nzd1 = mset1.NonzeroDiagonalEntries();
            nzd2 = mset2.NonzeroDiagonalEntries();
            testCase.verifyEqual( nzd1, nzd2 );
            testCase.verifyEqual( nzd1, mset2.Size());
        end
    end
end
