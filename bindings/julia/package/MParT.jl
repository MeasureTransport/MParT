# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    @wrapmodule("libmpartjl")

    function __init__()
        @initcxx
    end
end