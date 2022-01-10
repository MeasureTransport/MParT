#include <catch2/catch_test_macros.hpp>

#include "MParT/MultiIndexSet.h"


TEST_CASE( "Testing the MultiIndeSet class", "[MultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    mpart::MultiIndexSet mset(dim,maxOrder);

    REQUIRE( mset.NumTerms()==((maxOrder+1)*(maxOrder+2)/2));
    
}