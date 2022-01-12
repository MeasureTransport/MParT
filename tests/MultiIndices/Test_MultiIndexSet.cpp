#include <catch2/catch_test_macros.hpp>

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"

using namespace mpart;

TEST_CASE( "Testing the FixedMultiIndexSet class", "[FixedMultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    FixedMultiIndexSet mset(dim,maxOrder);

    REQUIRE( mset.NumTerms()==((maxOrder+1)*(maxOrder+2)/2));
}



TEST_CASE("Testing the MultiIndexSet class", "[MultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

    REQUIRE( mset.Size()==((maxOrder+1)*(maxOrder+2)/2));

}