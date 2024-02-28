classdef SigmoidTest < matlab.unittest.TestCase
    % Copyright 2014 - 2016 The MathWorks, Inc.

    methods ( Test )

        function SigmoidComponent( testCase )
            num_sigmoids = 5;
            input_dim = 9;
            centers = createSigmoidCenters( num_sigmoids );
            num_sigmoids_order = 2;
            max_order = num_sigmoids_order+2;
            opts = MapOptions;
            opts.basisType = BasisTypes.HermiteFunctions;
            comp = CreateSigmoidComponent(input_dim, max_order, centers, opts);
            expected_coeffs = nchoosek(input_dim+max_order, max_order);
            testCase.verifyEqual( comp.numCoeffs, uint32(expected_coeffs) );
        end

        function SigmoidComponentMset( testCase )
            num_sigmoids = 5;
            input_dim = 9;
            centers = createSigmoidCenters( num_sigmoids );
            num_sigmoids_order = 2;
            max_order = num_sigmoids_order+2;
            opts = MapOptions;
            opts.basisType = BasisTypes.HermiteFunctions;
            mset_off = FixedMultiIndexSet(input_dim-1, max_order);
            mset_diag = MultiIndexSet.CreateNonzeroDiagTotalOrder(input_dim, max_order).Fix();
            comp = CreateSigmoidComponent(mset_off, mset_diag, centers, opts);
            expected_coeffs = mset_off.Size() + mset_diag.Size();
            testCase.verifyEqual( comp.numCoeffs, uint32(expected_coeffs) );
        end

        function SigmoidTriangular( testCase )
            num_sigmoids = 5;
            input_dim = 9;
            output_dim = 3;
            centers = createSigmoidCenters( num_sigmoids );
            centers_total = repmat(centers, 1, output_dim);
            num_sigmoids_order = 2;
            max_order = num_sigmoids_order+2;
            opts = MapOptions;
            opts.basisType = BasisTypes.HermiteFunctions;
            comp = CreateSigmoidTriangular(input_dim, output_dim, max_order, centers_total, opts);
            expected_coeffs = sum(arrayfun(@(d) nchoosek(d+max_order, max_order), (input_dim-output_dim+1):input_dim));
            testCase.verifyEqual( comp.numCoeffs, uint32(expected_coeffs) );
        end
    end
end

function centers = createSigmoidCenters( num_sigmoids, bound )
    if nargin < 2
        bound = 3;
    end
    num_params = 2 + num_sigmoids*(num_sigmoids+1)/2;
    centers = zeros(num_params,1);
    centers(1) = -bound;
    centers(2) = bound;
    centers_idx = 3;
    for order = 1:num_sigmoids
        for i = 1:order
            centers(centers_idx) = -bound + (i-1)*2*bound/(order-1);
            centers_idx = centers_idx + 1;
        end
    end
end
