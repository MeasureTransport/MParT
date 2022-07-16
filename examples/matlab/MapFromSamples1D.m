clear all
close all

addpath(genpath('~/Installations/MParT/matlab/'))

addpath(genpath('.'));

KokkosInitialize(8);

% training samples
nsamp = 5000;
z = randn(1,nsamp);
x = sinharcsinh(z,-1,1,0.5,1);


multis = 0:5; %need to specify it as (order,dim) !!!
mset = MultiIndexSet(multis');

opts = MapOptions(); %Setting options to default
map = ConditionalMap(mset,opts); %create ConditionalMap with Matlab MultiIndexSet and MapOptions


ref = Normal1D();

x_ref = linspace(-4,4,100);
figure
fig = gcf;
fig.Color = [1 1 1];
subplot(1,2,1)
histogram(x,50,'Normalization','pdf')
hold on
plot(x_ref,exp(ref.LogPdf(x_ref)),'LineWidth',1.5)
ax = gca;
ax.FontSize = 14;
legend('initial samples','normal pdf')

optimize_KL(map,ref,x);
z_tm = map.Evaluate(x);
subplot(1,2,2)
histogram(z_tm,50,'Normalization','pdf')
hold on
plot(x_ref,exp(ref.LogPdf(x_ref)),'LineWidth',1.5)
ax = gca;
ax.FontSize = 14;
legend('transported samples','normal pdf')
