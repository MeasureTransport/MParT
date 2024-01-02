
set (MPART_SERIALIZE_TESTS "")
set (MPART_OPT_TESTS "")
if (MPART_ARCHIVE)
     set (MPART_SERIALIZE_TESTS tests/Test_Serialization.cpp)
endif ()
if (MPART_OPT)
     set (MPART_OPT_TESTS
          tests/Test_MapObjective.cpp
          tests/Test_TrainMap.cpp
          tests/Test_TrainMapAdaptive.cpp
     )
endif ()

set (TEST_SOURCES
     tests/RunTests.cpp

     # MultiIndex tests
     tests/MultiIndices/Test_MultiIndex.cpp
     tests/MultiIndices/Test_MultiIndexSet.cpp

     # Distributions tests
     tests/Distributions/Test_Distributions_Common.cpp
     tests/Distributions/Test_DensityBase.cpp
     tests/Distributions/Test_SampleGenerator.cpp
     tests/Distributions/Test_Distribution.cpp
     tests/Distributions/Test_GaussianDistribution.cpp
     tests/Distributions/Test_TransportDensity.cpp
     tests/Distributions/Test_TransportSampler.cpp

     tests/Test_OrthogonalPolynomials.cpp
     tests/Test_RootFinding.cpp
     tests/Test_HermiteFunctions.cpp
     tests/Test_Sigmoid.cpp
     tests/Test_PositiveBijectors.cpp
     tests/Test_Quadrature.cpp
     tests/Test_MonotoneComponent.cpp
     tests/Test_MultivariateExpansion.cpp
     tests/Test_BasisEvaluator.cpp
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

     ${MPART_SERIALIZE_TESTS}
     ${MPART_OPT_TESTS}
PARENT_SCOPE)
