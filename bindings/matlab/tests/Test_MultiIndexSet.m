clear;
addpath(genpath('.'));

KokkosInitialize(8);

multis1 = [0 1 2 3]';
multis2 = [0 1;2 0]; %need to specify it as (order,dim) !!!
mset1 = MultiIndexSet(multis1);
mset2 = MultiIndexSet(multis2);


%Visualize
disp('Visualize: ')
mset2.Visualize()

disp('Max Orders: ')
disp(mset2.MaxOrders())

disp('Size: ')
disp(mset2.Size())

disp('Expand: ')
disp(mset2.Expand(2))
mset2.Visualize()

disp('ForciblyExpand: ')
disp(mset2.ForciblyExpand(3))
disp('Size: ')
disp(mset2.Size())

disp('Frontier: ')
disp(mset2.Frontier())

disp('Strict Frontier: ')
disp(mset2.Frontier())

disp('BackwardNeighbors: ')
disp(mset2.BackwardNeighbors(2))

disp('IsExpandable: ')
disp(mset2.IsExpandable(1))

disp('NumActiveForward: ')
disp(mset2.NumActiveForward(1))


disp('NumForward: ')
disp(mset2.NumForward(1))

multis = [0 1;2 0];
mset2 = MultiIndexSet(multis);

disp('IndexToMulti: ')
multi = mset2.IndexToMulti(2);
disp(multi.Vector())

disp('MultiToIndex: ')
idx = mset2.MultiToIndex(multi);
disp(idx)

disp('at: ')
multi = mset2.at(idx);
disp(multi.Vector)

disp('{} operator: ')
multi = mset2{idx};
disp(multi.Vector())

disp('+ multiIndex: ')
mset = MultiIndexSet([1 3;3 3]);
mset.Visualize()
a=[0,1];
multiAdd = MultiIndex(a);
mset+multiAdd;
mset.Visualize()


disp('+ multiIndexSet: ')
msetAdd = MultiIndexSet([1 1]);

a=[2,3];
multi=MultiIndex(a);
mset+multi;
mset.Visualize();

disp('Union:')
mset = MultiIndexSet([0 0;1 1]);
mset2 = MultiIndexSet([1 1;2 2]);
mset.Union(mset2);
mset.Visualize()

disp('Activate')
a=[1,1];
multi=MultiIndex(a);
mset.Activate(multi)
mset.Visualize()

disp('AddActive')
a=[3,3];
multi=MultiIndex(a);
mset.AddActive(multi)

disp('ForciblyActivate')
a=[2,0];
multi=MultiIndex(a);
mset.ForciblyActivate(multi)

disp('AdmissibleFowardNeighbors')
listMultis = mset.AdmissibleForwardNeighbors(1);
for k=1:length(listMultis)
    disp(listMultis(k).String)
end

mset.Visualize()

disp('Margin')
listMultis = mset.Margin();
for k=1:length(listMultis)
    disp(listMultis(k).String)
end


disp('ReducedMargin')
listMultis = mset.ReducedMargin();
for k=1:length(listMultis)
    disp(listMultis(k).String)
end

disp('ReducedMarginDim')
listMultis = mset.ReducedMarginDim(1);
for k=1:length(listMultis)
    disp(listMultis(k).String)
end