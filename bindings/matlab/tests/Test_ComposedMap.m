clear

addpath(genpath('~/Installations/MParT/'));

KokkosInitialize(8);

opts = MapOptions(); %Setting options to default

T1 = CreateTriangular(2,2,2,opts);
T2 = CreateTriangular(2,2,2,opts);

T = ComposedMap([T1 T2]);

Y=T.Evaluate(randn(2,10));
