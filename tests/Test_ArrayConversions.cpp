#include <catch2/catch_all.hpp>

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

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


#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

TEST_CASE( "Testing functions that copy views between host and device", "[ArrayConversionsHostDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    unsigned int N1 = 10;
    unsigned int N2 = 20;

    Kokkos::View<double*, Kokkos::HostSpace> hostVec("host stuff", N1);
    for(unsigned int i=0; i<N1; ++i)
        hostVec(i) = i;
    
    // Copy to the device 
    auto deviceVec = ToDevice<DeviceSpace>(hostVec);

    // Copy back to host 
    auto hostVec2 = ToHost(deviceVec);
    REQUIRE(hostVec2.extent(0)==N1);
    for(unsigned int i=0; i<N1; ++i)
        CHECK( hostVec2(i) ==i );

    // Copy a slice back to host 
    auto slice1 = ToHost(deviceVec, std::make_pair(1,3));
    REQUIRE( slice1.extent(0) == 2);
    CHECK(slice1(0)==1);
    CHECK(slice1(1)==2);
}


TEST_CASE( "Testing mapping from memory space to valid execution space.", "[KokkosSpaceMapping]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    unsigned int N1 = 10;

    Kokkos::View<double*, Kokkos::HostSpace> hostVec("host stuff", N1);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space>(0,N1), KOKKOS_LAMBDA (const int i) {
        hostVec(i) = i;
    });
    
    for(unsigned int i=0; i<N1; ++i)
        CHECK( hostVec(i) ==i );

    // Copy to the device 
    Kokkos::View<double*,DeviceSpace> deviceVec("device stuff", N1);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space>(0,N1), KOKKOS_LAMBDA (const int i) {
        deviceVec(i) = i;
    });

    // Copy the device vector back to host and compare
    auto hostVec2 = ToHost(deviceVec);
    REQUIRE(hostVec2.extent(0)==N1);
    for(unsigned int i=0; i<N1; ++i)
        CHECK( hostVec2(i) ==i );
}


#endif 