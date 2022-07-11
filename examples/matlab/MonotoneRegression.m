clear all
close all

addpath(genpath('~/Installations/MParT/matlab/')) %installation path

addpath(genpath('.'));

KokkosInitialize(8);

nsamp = 500;
y = [zeros(1,nsamp) 2*ones(1,nsamp)] + .4*randn(1,2*nsamp)  ;
x = linspace(0,4,2*nsamp);

order=5;
multis = 0:order;
mset = MultiIndexSet(multis');

opts = MapOptions(); %Setting options to default
opts.quadMinSub=4;

map = ConditionalMap(mset,opts); %create ConditionalMap with Matlab MultiIndexSet and MapOptions


figure
fig = gcf;
fig.Color = [1 1 1];
subplot(1,2,1)
plot(x,y,'--*')
hold on
plot(x,map.Evaluate(x),'LineWidth',2)
legend('training samples','map approximation')
error0 = objective_LS(map,x,y);
title(['Initial regression error: ',num2str(error0)])
ylim([-1.5 4])
optimize_LS(map,x,y); 
ax = gca;
ax.FontSize = 14;
error = objective_LS(map,x,y);

subplot(1,2,2)
plot(x,y,'--*')
hold on
plot(x,map.Evaluate(x),'LineWidth',2)
title(['Final regression error: ',num2str(error)])
legend('training samples','map approximation')
ylim([-1.5 4])
ax = gca;
ax.FontSize = 14;