function map = CreateSigmoidTriangular(inDim, outDim, totalOrder, centers, opts)
    map = ConditionalMap(inDim, outDim, totalOrder, centers, opts);
end