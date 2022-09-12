target_sources(mpart 
    PRIVATE
    MultiIndices/MultiIndex.cpp
    MultiIndices/MultiIndexLimiter.cpp
    MultiIndices/MultiIndexSet.cpp
    MultiIndices/FixedMultiIndexSet.cpp
    MultiIndices/MultiIndexNeighborhood.cpp

    Utilities/Miscellaneous.cpp

    ParameterizedFunctionBase.cpp
    ConditionalMapBase.cpp
    TriangularMap.cpp
    MapFactory.cpp
    MultivariateExpansionWorker.cpp

    Initialization.cpp
)