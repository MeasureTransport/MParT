clear

addpath(genpath('~/Installations/MParT/'));

KokkosInitialize(8);

A = 2.*eye(2);
b = [-0.3;0.5];

T1 = AffineMap(A);
T2 = AffineMap(b);
T3 = AffineMap(A,b);

x = [linspace(-4,4,11);linspace(-4,4,11)];

y1 = T1.Evaluate(x);
y2 = T2.Evaluate(x);
y3 = T3.Evaluate(x);

