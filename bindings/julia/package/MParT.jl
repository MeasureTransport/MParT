# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    @wrapmodule(joinpath(".","libmpartjl"))

    function __init__()
        @initcxx
        Initialize()
    end

    export SetCoeffs, MapOptions, MultiIndexSet,
           Fix, CoeffMap, LogDeterminant, CreateComponent,
           Evaluate, to_base
end