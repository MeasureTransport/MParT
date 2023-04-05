
if (MPART_OPT)
    set(MPART_OPT_FILES
        MapObjective.cpp
        TrainMap.cpp
        AdaptiveTransportMap.cpp
    )
else()
    set(MPART_OPT_FILES "")
endif ()

target_sources(mpart
    PRIVATE
    MultiIndices/MultiIndex.cpp
    MultiIndices/MultiIndexLimiter.cpp
    MultiIndices/MultiIndexSet.cpp
    MultiIndices/FixedMultiIndexSet.cpp
    MultiIndices/MultiIndexNeighborhood.cpp

    Utilities/Miscellaneous.cpp
    Utilities/LinearAlgebra.cpp

    Distributions/DensityBase.cpp
    Distributions/GaussianSamplerDensity.cpp
    Distributions/PullbackDensity.cpp

    ParameterizedFunctionBase.cpp
    ConditionalMapBase.cpp
    TriangularMap.cpp
    IdentityMap.cpp
    ComposedMap.cpp
    SummarizedMap.cpp
    # DebugMap.cpp
    AffineMap.cpp
    AffineFunction.cpp
    MapFactory.cpp

    MapFactoryImpl1.cpp
    MapFactoryImpl2.cpp
    MapFactoryImpl3.cpp
    MapFactoryImpl4.cpp
    MapFactoryImpl5.cpp
    MapFactoryImpl6.cpp
    MapFactoryImpl7.cpp
    MapFactoryImpl8.cpp
    MapFactoryImpl9.cpp
    MapFactoryImpl10.cpp
    MapFactoryImpl11.cpp
    MapFactoryImpl12.cpp
    MapFactoryImpl13.cpp
    MapFactoryImpl14.cpp
    MapFactoryImpl15.cpp

    ${MPART_OPT_FILES}
    Initialization.cpp
)