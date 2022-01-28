#include <catch2/catch_all.hpp>

#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Pointer to Kokkos Conversions in 1D", "[ArrayConversions1D]" ) {

    unsigned int length = 10;

    SECTION("double"){

        // Initialize a std vector
        std::vector<double> data(length);
        for(unsigned int i=0; i<length; ++i)
            data[i] = i;
        

        auto view = ToKokkos(&data[0], length);
        REQUIRE(view.extent(0) == length);
        for(unsigned int i=0; i<length; ++i){

            // Make sure the values are the same
            CHECK( data[i] == view(i) );

            // Check sure the memory address is the same  (i.e., we're not copying)
            CHECK( &data[i] == &view(i) );
        }
    }


    SECTION("int"){

        // Initialize a std vector
        std::vector<int> data(length);
        for(unsigned int i=0; i<length; ++i)
            data[i] = i;
        
        auto view = ToKokkos(&data[0], length);
        REQUIRE(view.extent(0) == length);
        for(unsigned int i=0; i<length; ++i){

            // Make sure the values are the same
            CHECK( data[i] == view(i) );

            // Check sure the memory address is the same  (i.e., we're not copying)
            CHECK( &data[i] == &view(i) );
        }
    }

}

TEST_CASE( "Testing Pointer to Kokkos Conversions in 2D", "[ArrayConversions2D]" ) {

    unsigned int rows = 10;
    unsigned int cols = 20;


    SECTION("double"){

        // Initialize a std vector
        std::vector<double> data(rows*cols);
        for(unsigned int i=0; i<rows*cols; ++i)
            data[i] = i;
        

        auto rowView = ToKokkos<double, Kokkos::LayoutRight>(&data[0], rows, cols);

        REQUIRE(rowView.extent(0) == rows);
        REQUIRE(rowView.extent(1) == cols);

        for(unsigned int i=0; i<rows; ++i){
            for(unsigned int j=0; j<cols; ++j){
                // Make sure the values are the same
                CHECK( data[i*cols+j] == rowView(i,j) );

                // Check sure the memory address is the same  (i.e., we're not copying)
                CHECK( &data[i*cols+j] == &rowView(i,j) );
            }
        }

        auto colView = ToKokkos<double, Kokkos::LayoutLeft>(&data[0], rows, cols);

        REQUIRE(rowView.extent(0) == rows);
        REQUIRE(rowView.extent(1) == cols);

        for(unsigned int i=0; i<rows; ++i){
            for(unsigned int j=0; j<cols; ++j){
                // Make sure the values are the same
                CHECK( data[i +j*rows] == colView(i,j) );

                // Check sure the memory address is the same  (i.e., we're not copying)
                CHECK( &data[i + j*rows] == &colView(i,j) );
            }
        }
    }


}