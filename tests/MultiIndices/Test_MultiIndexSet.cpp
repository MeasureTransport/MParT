#include <catch2/catch_test_macros.hpp>

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"

using namespace mpart;

TEST_CASE( "Testing the FixedMultiIndexSet class", "[FixedMultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    FixedMultiIndexSet mset(dim,maxOrder);

    REQUIRE( mset.Size()==((maxOrder+1)*(maxOrder+2)/2));
}


TEST_CASE("Conversions between MultiIndexSet types", "[MultiIndexSet Conversions]" ) {

    unsigned int dim = 10;
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, 3);

    FixedMultiIndexSet fixedSet = mset.Compress();

    REQUIRE(mset.Size() == fixedSet.Size() );

    // Make sure the active multiindices are exactly the same
    for(unsigned int i=0; i<mset.Size(); ++i){
        std::vector<unsigned int> fixedVec = fixedSet.IndexToMulti(i);
        std::vector<unsigned int> vec = mset.IndexToMulti(i).Vector();

        for(unsigned int d=0; d<dim; ++d)
            REQUIRE(fixedVec.at(d)==vec.at(d));
    }
}

TEST_CASE("Testing the MultiIndexSet class", "[MultiIndexSet]" ) {

    // const unsigned int dim = 2;
    // const unsigned int maxOrder = 5;

    // MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

    // REQUIRE( mset.Size()==((maxOrder+1)*(maxOrder+2)/2));


    /*
    AdmissableNeighbor.ValidNeighbor
    --------------------------------

    Purpose:
    Make sure the MultiIndex class will return valid admissable neighbors.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 1. Add an admissable
    neighbor to index (2,0), and check to make sure that index is an admissable
    neighbor to the MultiIndexSet.

    3 |                         3 |
    2 |                         2 |
    1 | x   x                   1 | x   x
    0 | x   x   o               0 | x   x   x
        ----------------            ----------------
        0   1   2   3               0   1   2   3
    */
    SECTION("ValidNeighbor")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

        // MultiIndex for testing.
        MultiIndex multi{2,0};

        bool isAdmiss = indexFamily.IsAdmissible(multi);
        // Check the result of IsAdmissable().
        REQUIRE( indexFamily.IsAdmissible(multi) );
    }

    SECTION("Visualize")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTotalOrder(2, 3);

        std::stringstream truth, output;
        truth << " 4 | o  \n 3 | x  o  \n 2 | x  x  o  \n 1 | x  x  x  o  \n 0 | x  x  x  x  o  \n    ----------------\n     0  1  2  3  4  \n";

        indexFamily.Visualize(output);

        REQUIRE(output.str() == truth.str());
    }

    // /*
    // AdmissableNeighbor.UndefinedNeighbor
    // ------------------------------------

    // Purpose:
    // Make sure that MultiIndex class will not return a valid neighbor for indices
    // that do not have defined admissable neighbors.

    // Test:
    // Create MultiIndexSet with dimension of 2 and max order of 1. Add an admissable
    // neighbor to index (2,1) and, check to make sure that index is not returned as
    // an admissable neighbor to the MultiIndexSet - missing a required neighboring
    // index (2,0).

    // 3 |                          3 |
    // 2 |                          2 |
    // 1 | x   x   o                1 | x   x   o
    // 0 | x   x                    0 | x   x
    //     -----------------            -----------------
    //     0   1   2   3                0   1   2   3
    // */
    // SECTION("UndefinedNeighbor")
    // {
    //     // MultiIndexSet - the "square".
    //     MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

    //     // Create MultiIndex for testing against the square.
    //     MultiIndex multi{2,1};

    //     // Check the result of IsAdmissable().
    //     REQUIRE( !indexFamily.IsAdmissible(multi));
    // }

    // /*
    // AdmissableNeighbor.OutsideMaxOrder
    // ----------------------------------

    // Purpose:
    // Make sure that MultiIndex class will not return a valid neighbor for indices
    // that are outside the max order limit.

    // Test:
    // Create MultiIndexSet with dimension of 2 and order of 3, and create a max
    // index limiter of 3. Add an admissable neighbor to index (4,0), and check to
    // make sure that index is not returned as an admissable neighbor to the
    // MultiIndexSet.

    // 4 |                           4 |
    //   -----------------             -----------------
    // 3 | x   x   x   x |           3 | x   x   x   x |
    // 2 | x   x   x   x |           2 | x   x   x   x |
    // 1 | x   x   x   x |           1 | x   x   x   x |
    // 0 | x   x   x   x | o         0 | x   x   x   x | o
    //   --------------------          --------------------
    //     0   1   2   3   4             0   1   2   3   4

    // */
    // SECTION("OutsideMaxOrder")
    // {
    //     // Max order limit of 3.
    //     MultiIndexLimiter::MaxDegree limiter(std::vector<unsigned int>{3,3});

    //     // MultiIndexSet - the "square".
    //     MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 3, limiter);

    //     // Create MultiIndex for testing against the square.
    //     MultiIndex multi{4,0};

    //     // Check the result of IsAdmissable().
    //     REQUIRE( !indexFamily.IsAdmissible(multi) );
    // }

    // /*
    // AdmissableNeighbor.OutsideTotalOrder
    // ------------------------------------

    // Purpose:
    // Make sure that MultiIndex class will not return a valid neighbor for indices
    // that are outside the total order.

    // Test:
    // Create MultiIndexSet with dimension of 2 and max order of 3, and create a
    // total index limiter of 3. Add two admissable neighbors to indices (2,2) and
    // (4,0), and check to make sure that both indices are not returned as
    // admissable neighbors to the MultiIndexSet.

    // 4 |                           4 |
    // 3 | x\                        3 | x\
    // 2 | x   x\  o                 2 | x   x\  o
    // 1 | x   x   x\                1 | x   x   x\
    // 0 | x   x   x   x\  o         0 | x   x   x   x\  o
    //     --------------------          --------------------
    //     0   1   2   3   4             0   1   2   3   4
    // */
    // SECTION("OutsideTotalOrder")
    // {
    //     // Total order limit of 3.
    //     MultiIndexLimiter::TotalOrder limiter(3);

    //     // MultiIndexSet - the "square".
    //     MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 3, limiter);

    //     // Create MultiIndex for testing against the square.
    //     MultiIndex multi1{4,0};
    //     MultiIndex multi2{2,2};

    //     // Check the result of IsAdmissable().
    //     REQUIRE( !indexFamily.IsAdmissible(multi1));

    //     REQUIRE( !indexFamily.IsAdmissible(multi2));
    // }

    // /*
    // AdmissableNeighbor.AddAdmissibleNeighbor
    // ----------------------------------------

    // Purpose:
    // Make sure that MultiIndex class will return a valid neighbor for admissible
    // neighbors that have been added to the MultiIndexSet.

    // Test:
    // Create MultiIndexSet with dimension of 2 and max order of 1, add an admissible
    // neighbor at (2,0), then add an admissable neighbor to index (2,1). Then check
    // to make sure that new index is returned as an admissable neighbor to the
    // MultiIndexSet.

    // 3 |                          3 |
    // 2 |                          2 |
    // 1 | x   x   o                1 | x   x   x
    // 0 | x   x   x                0 | x   x   x
    //     ----------------             ----------------
    //     0   1   2   3                0   1   2   3
    // */
    // SECTION("AddAdmissibleNeighbor")
    // {
    //     // MultiIndexSet - the "square".
    //     MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

    //     // Add forward admissible neighbor to index (1,0).
    //     MultiIndex newIndex{2,0};
    //     indexFamily.AddActive(newIndex);

    //     // Create MultiIndex for testing against the square.
    //     MultiIndex multi{2,1};

    //     // Check the result of IsAdmissable().
    //     REQUIRE(indexFamily.IsAdmissible(multi));
    // }


    // SECTION("BackwardsNeighbors")
    // {

    //     MultiIndexSet indexSet = MultiIndexSet::CreateTensorProduct(2, 3);
    //     MultiIndex newMulti{2,1};

    //     std::vector<unsigned int> neighbors = indexSet.GetBackwardNeighbors(indexSet.MultiToIndex(newMulti));

    //     REQUIRE(neighbors.size() == 2);

    //     std::vector<unsigned int> neigh1 = indexSet.IndexToMulti(neighbors.at(0)).Vector();
    //     std::vector<unsigned int> neigh2 = indexSet.IndexToMulti(neighbors.at(1)).Vector();

    //     REQUIRE(1 == neigh1.at(0));
    //     REQUIRE(1 == neigh1.at(1));
    //     REQUIRE(2 == neigh2.at(0));
    //     REQUIRE(0 == neigh2.at(1));
    // }

    // /*
    // AdmissableNeighbor.ForciblyExpandAdmissibleNeighbors
    // ----------------------------------------------------

    // Purpose:
    // Make sure that MultiIndex class will return a valid neighbor for admissible
    // neighbors that have been added to the MultiIndexSet.

    // Test:
    // Create MultiIndexSet with dimension of 2 and max order of 1, add admissible
    // neighbors at (1,1) using the ForciblyExpand function. Add an admissable
    // neighbor to index (3,0), and check to make sure that index is returned as an
    // admissable neighbor to the MultiIndexSet.

    // 3 |                         3 |
    // 2 | o   o   o               2 | o   o   o
    // 1 | x   x   o               1 | x   x   o   o
    // 0 | x   x   o               0 | x   x   x   o
    //     -----------------           -------------------
    //     0   1   2   3               0   1   2   3   4
    // */
    // SECTION("Expand")
    // {
    //     // MultiIndexSet - the "square".
    //     MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

    //     // Add forward admissible neighbor to index (1,1) using ForciblyExpand.
    //     MultiIndex newIndex{1,0};

    //     int oldSize = indexFamily.Size();
    //     REQUIRE(oldSize==4);

    //     int activeIndex = indexFamily.MultiToIndex(newIndex);
    //     REQUIRE(activeIndex>=0);
        
    //     indexFamily.Expand(activeIndex);
    //     REQUIRE( (oldSize+1) == indexFamily.Size());

    //     // Create MultiIndex for testing against the square.
    //     MultiIndex multi{3,0};

    //     // Check the result of IsAdmissable().
    //     REQUIRE( indexFamily.IsAdmissible(multi));
    // }
}