clear all
close all

addpath(genpath('~/Installations/MParT/matlab/')) %installation path
addpath(genpath('.'));

num_threads = 8;
KokkosInitialize(num_threads); 

% training samples
nsamp = 5000;
X = sample_banana(nsamp);
%X = load('BananaSamples_norm.txt')';

opts = MapOptions(); %Setting options to default

multis1 = [0 1]';
multis2 = [0 1;2 0]; 
mset1 = MultiIndexSet(multis1);
mset2 = MultiIndexSet(multis2);

map1 = ConditionalMap(mset1,opts); %create ConditionalMap with Matlab MultiIndexSet and MapOptions
map2 = ConditionalMap(mset2,opts);

ref = Normal1D();

optimize_KL(map1,ref,X(1,:));
optimize_KL(map2,ref,X);

Y(:,1) = map1.Evaluate(X(1,:));
Y(:,2) = map2.Evaluate(X);

x1 = linspace(-4,4,100);
x2 = linspace(-4,4,100);
[x,y] = meshgrid(x1,x2);
xy = [x(:) y(:)];

X_ref = mvnpdf(xy);

figure
fig = gcf;
fig.Color = [1 1 1];
subplot(1,2,1)
contour(x,y,reshape(X_ref,100,100),'LineWidth',1.5)
hold on
plot(X(1,1:1000)',X(2,1:1000)','r*')
ax = gca;
ax.FontSize = 14;
legend('normal pdf','1000 training samples')

subplot(1,2,2)
contour(x,y,reshape(X_ref,100,100),'LineWidth',1.5)
hold on
plot(Y(1:1000,1),Y(1:1000,2),'r*')
ax = gca;
ax.FontSize = 14;
legend('normal pdf','1000 transported training samples')


%% -- DEFINE MODEL --

function X = sample_banana(N)
    x1 = randn(N,1);
    x2 = x1.^2 + randn(N,1);
    X = [x1, x2];
    X=X';
end