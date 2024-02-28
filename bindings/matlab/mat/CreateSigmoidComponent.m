function map = CreateSigmoidComponent(inputDim, totalOrder, centers, options)
    map = ConditionalMap(inputDim, totalOrder, centers, options);
end