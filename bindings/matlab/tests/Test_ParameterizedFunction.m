clear
addpath(genpath('~/Installations/MParT/matlab/'))

addpath(genpath('.'));

KokkosInitialize(8);

opts = MapOptions(); %Setting options to default

multis2 = [0 1;2 0];
mset2 = MultiIndexSet(multis2);

ParFunc = ParameterizedFunction(2,mset2,opts);
ParFunc.SetCoeffs(ones(1,ParFunc.numCoeffs))

disp('Coeffs')
disp(ParFunc.Coeffs)

disp('Evaluate')
disp(ParFunc.Evaluate(randn(2,10)))
