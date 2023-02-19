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
            obj1 = GaussianKLObjective(train, test, 1);
            obj2 = GaussianKLObjective(train, test);
            map_options = MapOptions();
            max_order = 2;
            map1 = CreateComponent(FixedMultiIndex(dim,max_order),map_options);
            map2 = CreateTriangular(dim,dim,max_order,map_options);
            map1.SetCoeffs(zeros(map1.numCoeffs,1));
            map2.SetCoeffs(zeros(map2.numCoeffs,1));

            % Set Training Options
            train_options = TrainOptions;

            % Print test error before
            TrainMap(map1, obj1, train_options);
            TrainMap(map2, obj2, train_options);

            % Evaluate test samples after training
            KS_stat1 = KSStatistic(map1,test);
            KS_stat2 = KSStatistic(map2,test);
            testCase.verifyTrue(KS_stat1 < 0.1);
            testCase.verifyTrue(KS_stat2 < 0.1);
        end
    end
end


% Perform Kolmogorov-Smirnov test
function KS_stat = KSStatistic(map,samples)
    pullback_evals = Evaluate(map,samples);
    sorted_samples = sort(pullback_evals(:));
    samps_cdf = (1 + erf(sorted_samples/sqrt(2)))/2;
    samps_ecdf = (1:numel(samples))'/numel(samples);
    KS_stat = max(abs(samps_ecdf - samps_cdf));
end