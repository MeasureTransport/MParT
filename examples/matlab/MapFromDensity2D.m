addpath(genpath('~/Installations/MParT/matlab/')) %installation path

addpath(genpath('.'));

KokkosInitialize(8);

opts = MapOptions(); %Setting options to default
% opts.contDeriv = false;
% opts.quadMinSub = 3;

multis1 = [0 1]';
multis2 = [0 1;2 0]; %need to specify it as (order,dim) !!!
mset1 = MultiIndexSet(multis1);
mset2 = MultiIndexSet(multis2);

map1 = ConditionalMap(mset1,opts); %create ConditionalMap with Matlab MultiIndexSet and MapOptions
map2 = ConditionalMap(mset2,opts);

triMap = TriangularMap([map1.get_id(),map2.get_id()]);
triMap.SetCoeffs(zeros(1,triMap.numCoeffs))

ref = Banana();

xx = linspace(-3,3,100);
yy = linspace(-3,3,100);
[X1,X2] = meshgrid(xx,yy);
logpi_true = ref.LogPdf([X1(:),X2(:)]);
logpi = reshape(logpi_true,100,100);

figure
fig=gcf;
fig.Color = [1 1 1];
subplot(1,2,1)
contour(xx,yy,exp(logpi),10,'LineWidth',1.5)
hold on
N = 5000;
Z = randn(2,N);
plot(Z(1,1:2000),Z(2,1:2000),'r*')
legend('target pdf','training samples')

optimize_KL(triMap,ref,Z);

subplot(1,2,2)
Y = triMap.Evaluate(Z);
contour(xx,yy,exp(logpi),10,'LineWidth',1.5)
hold on
plot(Y(1,1:2000),Y(2,1:2000),'r*')
legend('target pdf','transported samples')

