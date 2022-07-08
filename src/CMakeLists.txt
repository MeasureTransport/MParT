target_sources(mpart 
    PRIVATE
    MultiIndices/MultiIndex.cpp
    MultiIndices/MultiIndexLimiter.cpp
    MultiIndices/MultiIndexSet.cpp
    MultiIndices/FixedMultiIndexSet.cpp
    MultiIndices/MultiIndexNeighborhood.cpp

    Utilities/Miscellaneous.cpp

    ConditionalMapBase.cpp
    TriangularMap.cpp
    MapFactory.cpp

    Initialization.cpp
)