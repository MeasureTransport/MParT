# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    @wrapmodule(joinpath(".","libmpartjl"))

    function __init__()
        @initcxx
    end
    Initialize()

    # macro __hasHead(x)
        
    # macro __CondMaptoBaseFcn(fcn,args...)
    #     argstup = Tuple(args)
    #     pass_argstup = Tuple([arg.head == Symbol("::") ? arg.args[1] : arg])
    #     quote
    #         $(esc(fcn))(map::Union{CxxRef{<:ConditionalMapBase},
    #             CxxWrap.CxxWrapCore.SmartPointer{<:ConditionalMapBase}},
    #             $(map(esc, argstup)...))
    #             return $(esc(fcn))(to_base(map))
    

    export SetCoeffs, MapOptions, MultiIndexSet,
           Fix, CoeffMap, LogDeterminant, CreateComponent,
           Evaluate, to_base
end