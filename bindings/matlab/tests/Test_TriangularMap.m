clear

addpath(genpath('../../../'));

KokkosInitialize(8);

opts = MapOptions(); %Setting options to default
multis1 = [0 1 2 3]';
multis2 = [0 1;2 0]; %need to specify it as (order,dim) !!!
mset1 = MultiIndexSet(multis1);
mset2 = MultiIndexSet(multis2);

map1 = ConditionalMap(mset1,opts); %create ConditionalMap with Matlab MultiIndexSet and MapOptions
map2 = ConditionalMap(mset2,opts);

triMap = TriangularMap([map1,map2]);

disp(map1.Coeffs)
coeffs = 0.1*[1 2 3 4 5 6];
triMap.SetCoeffs(coeffs)
disp(triMap.Coeffs)
disp(map1.Coeffs)

disp('GetComponent:')
map1FromTri = triMap.GetComponent(1);
map2FromTri = triMap.GetComponent(2);

disp(map1FromTri.Coeffs)
disp(map2FromTri.Coeffs)

% Setting new coeff values
map1FromTri.SetCoeffs([1 1 1 1])

disp('GetBaseFunction:')
parFunc = map1FromTri.GetBaseFunction();
parFunc.SetCoeffs(map1FromTri.Coeffs);
disp(parFunc.Evaluate(randn(1,10)))


% Test construction with different dimensions

opts=MapOptions;
maxDegree = 2;

numBlocks = 3;
dim1 = 2;
dim2 = 4;
dim3 = 3;
dim = dim1 + dim2 + dim3;

M1 = CreateTriangular(dim1,dim1,maxDegree,opts);
M2 = CreateTriangular(dim1+dim2,dim2,maxDegree,opts);
M3 = CreateTriangular(dim1+dim2+dim3,dim3,maxDegree,opts);

triMap = TriangularMap([M1 M2 M3]);

triMap.SetCoeffs(randn(dim,triMap.numCoeffs));

disp(['Num coeffs triMap: ',num2str(triMap.numCoeffs)])
disp(['Sum coeffs maps: ',num2str(M1.numCoeffs+M2.numCoeffs+M3.numCoeffs)])

Y=triMap.Evaluate(randn(dim,10));

