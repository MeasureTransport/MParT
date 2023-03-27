classdef TrainMapTest < matlab.unittest.TestCase
    % Copyright 2014 - 2016 The MathWorks, Inc.

    methods ( Test )
        function train(testCase)
            % Create data
            dim = 2;
            N=20000;
            N_test = N/5;
            data = randn(dim,N);
            target = [data(1,:);data(2,:) + data(1,:).^2];
            test = target(:,1:N_test);
            train = target(:,N_test+1:end);

            % Create objective and map
            obj = GaussianKLObjective(train, test);
            map_options = MapOptions();
            max_order = 2;
            map = CreateTriangular(dim,dim,max_order,map_options);
            map.SetCoeffs(zeros(map.numCoeffs,1));

            % Set Training Options
            train_options = TrainOptions;
            train_options.verbose = 1;
            train_options.opt_alg = 'LD_SLSQP';

            % Print test error before
            TrainMap(map, obj, train_options);

            % Evaluate test samples after training
            pullback_evals = Evaluate(map,test);

            % Perform Kolmogorov-Smirnov test
            sorted_samples = sort(pullback_evals(:));
            samps_cdf = (1 + erf(sorted_samples/sqrt(2)))/2;
            samps_ecdf = (1:2*N_test)'/(2*N_test);
            KS_stat = max(abs(samps_ecdf - samps_cdf));
            testCase.verifyTrue(KS_stat < 0.1)
        end
    end
end