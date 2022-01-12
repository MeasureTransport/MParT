#include <catch2/catch_test_macros.hpp>

#include "MParT/MultiIndices/FixedMultiIndexSet.h"


TEST_CASE( "Testing the FixedMultiIndexSet class", "[MultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    mpart::FixedMultiIndexSet mset(dim,maxOrder);

    REQUIRE( mset.NumTerms()==((maxOrder+1)*(maxOrder+2)/2));

}