set (TEST_SOURCES
     tests/RunTests.cpp

     tests/MultiIndices/Test_MultiIndex.cpp
     tests/MultiIndices/Test_MultiIndexSet.cpp
     tests/Test_OrthogonalPolynomials.cpp
     tests/Test_HermiteFunctions.cpp
     tests/Test_PositiveBijectors.cpp
     tests/Test_Quadrature.cpp
     tests/Test_MonotoneComponent.cpp
     tests/Test_MultivariateExpansion.cpp
     tests/Test_MultivariateExpansionWorker.cpp
     tests/Test_ArrayConversions.cpp
     tests/Test_ConditionalMapBase.cpp
     tests/Test_TriangularMap.cpp
     tests/Test_ComposedMap.cpp
     tests/Test_MapFactory.cpp
     tests/Test_LinearizedBasis.cpp
     tests/Test_AffineMap.cpp
     tests/Test_LinearAlgebra.cpp
     tests/Test_AffineFunction.cpp
     tests/Test_IdentityMap.cpp
     tests/Test_SummarizedMap.cpp
     #tests/Test_TensorProductFunction.cpp
PARENT_SCOPE)

if (MPART_ARCHIVE)
     set (TEST_SOURCES ${TEST_SOURCES}  tests/Test_Serialization.cpp PARENT_SCOPE)
endif ()