# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    @wrapmodule("libmpartjl")

    function __init__()
        @initcxx
        Initialize()
    end

    MultiIndexSet(A::AbstractMatrix{<:Integer}) = MultiIndexSet(Cint.(collect(A)))

    export SetCoeffs, MapOptions, MultiIndexSet,
           Fix, CoeffMap, LogDeterminant, CreateComponent,
           Evaluate, to_base, numCoeffs, CoeffGrad, LogDeterminantCoeffGrad,
           CreateTriangular, BasisType!
end