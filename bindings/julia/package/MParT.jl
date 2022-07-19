# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    @wrapmodule(joinpath(".","libmpartjl"))

    function __init__()
        @initcxx
    end
    Initialize()

    export SetCoeffs, MapOptions, MultiIndexSet,
           Fix, CoeffMap, LogDeterminant, CreateComponent,
           Evaluate, to_base, numCoeffs, CoeffGrad, LogDeterminantCoeffGrad,
           CreateTriangular, BasisType!
end