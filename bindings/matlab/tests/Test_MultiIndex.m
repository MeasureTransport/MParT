clear;
addpath(genpath('.'));

KokkosInitialize(8);

a=[2,3,4];
multi=MultiIndex(a);
multi.String()

multi2=MultiIndex(6,1);
multi2.String()

multi3=MultiIndex([0,1,2,3]); %Not sure we want to keep this
multi3.String()

multi3.Set(1,9);
multi3.Vector();

disp(multi2==multi3);
disp(multi2~=multi2);

disp(multi2>multi3)

disp(multi2>=multi3)
disp(multi2<=multi2)
disp(multi3<multi3);



