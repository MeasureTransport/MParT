#include <catch2/catch_all.hpp>

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/EigenTypes.h"
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

TEST_CASE( "Testing Eigen to Kokkos Conversions in 1D", "[EigenArrayConversions1D]" ) {

    unsigned int size = 128;
    Eigen::VectorXd x(size);
    for(unsigned int i=0; i<size; ++i)
        x(i) = i;

    SECTION("contiguous"){
        auto x_view = VecToKokkos<double>(x);
        for(unsigned int i=0; i<size; ++i){
            CHECK(x_view(i)==x(i));   
            CHECK(&x_view(i) == &x(i));
        }
    }

    SECTION("strided"){
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<2>> xslice(x.data(), size/2);
        auto x_view = VecToKokkos<double>(xslice);
        for(unsigned int i=0; i<size/2; ++i){
            CHECK(x_view(i)==x(2*i));   
            CHECK(&x_view(i) == &x(2*i));
        }
    }

    SECTION("slice"){

        Eigen::MatrixXd x2(size,size);
        for(unsigned int j=0; j<size; ++j){
            for(unsigned int i=0; i<size; ++i){
                x2(i,j) = i + j*size;
            }
        }

        auto row_view  = VecToKokkos<double>(x2.row(10));
        for(unsigned int i=0; i<size; ++i){
            CHECK(row_view(i)==x2(10,i));
            CHECK(&row_view(i)==&x2(10,i));
        }

        auto col_view  = VecToKokkos<double>(x2.col(10));
        for(unsigned int i=0; i<size; ++i){
            CHECK(col_view(i)==x2(i,10));
            CHECK(&col_view(i)==&x2(i,10));
        }

    }
}


TEST_CASE( "Testing constant Eigen to Kokkos Conversions in 2D", "[ConstEigenArrayConversions2D]" ) {

    unsigned int rows = 64;
    unsigned int cols = 32;

    SECTION("Row Major"){
        Eigen::MatrixXd x(rows,cols);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                x(i,j) = i + j*rows;
            }
        }

        Kokkos::View<const double**, Kokkos::LayoutLeft,Kokkos::HostSpace> x_view = ConstColMatToKokkos<double>(x);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                CHECK(x_view(i,j)==x(i,j));   
                CHECK(&x_view(i,j) == &x(i,j));
            }
        }
    }

    SECTION("Column Major"){
        Eigen::RowMatrixXd x(rows,cols);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                x(i,j) = i + j*rows;
            }
        }

        Kokkos::View<const double**, Kokkos::LayoutRight,Kokkos::HostSpace> x_view = ConstRowMatToKokkos<double>(x);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                CHECK(x_view(i,j)==x(i,j));   
                CHECK(&x_view(i,j) == &x(i,j));
            }
        }
    }
}


TEST_CASE( "Testing Eigen to Kokkos Conversions in 2D", "[EigenArrayConversions2D]" ) {

    unsigned int rows = 64;
    unsigned int cols = 32;
    Eigen::MatrixXd x(rows,cols);
    for(unsigned int j=0; j<cols; ++j){
        for(unsigned int i=0; i<rows; ++i){
            x(i,j) = i + j*rows;
        }
    }

    SECTION("contiguous"){
        Kokkos::View<double**, Kokkos::LayoutLeft,Kokkos::HostSpace> x_view = MatToKokkos<double>(x);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                CHECK(x_view(i,j)==x(i,j));   
                CHECK(&x_view(i,j) == &x(i,j));
            }
        }
    }


    SECTION("block"){
        Kokkos::View<double**, Kokkos::LayoutStride,Kokkos::HostSpace> x_view = MatToKokkos<double>(x.block(2,3,10,10));
        for(unsigned int i=0; i<10; ++i){
            for(unsigned int j=0; j<10; ++j){
                CHECK(x_view(i,j)==x(2+i,3+j));   
                CHECK(&x_view(i,j) == &x(2+i,3+j));
            }
        }
    }

    SECTION("transpose"){

        auto x_view = MatToKokkos<double>(x.transpose());
        for(unsigned int i=0; i<10; ++i){
            for(unsigned int j=0; j<10; ++j){
                CHECK(x_view(i,j)==x(j,i));   
                CHECK(&x_view(i,j) == &x(j,i));
            }
        }
    }

    SECTION("rowmajor"){

        Eigen::RowMatrixXd x2(rows,cols);
        for(unsigned int j=0; j<cols; ++j){
            for(unsigned int i=0; i<rows; ++i){
                x2(i,j) = i + j*rows;
            }
        }
        
        auto x_view = MatToKokkos<double>(x2);
        for(unsigned int i=0; i<10; ++i){
            for(unsigned int j=0; j<10; ++j){
                CHECK(x_view(i,j)==x2(i,j));   
                CHECK(&x_view(i,j) == &x2(i,j));
            }
        }
    }
}

TEST_CASE( "Testing Kokkos to Eigen Conversions in 1D", "[KokkosToEigen1d]" ) {

    unsigned int size = 20;
    Kokkos::View<double*, Kokkos::HostSpace> view("View", size);
    for(unsigned int i=0; i<size; ++i)
        view(i) = i;

    auto map = KokkosToVec<double>(view);
    for(unsigned int i=0; i<size; ++i){
        CHECK(view(i) == map(i));
        view(i) += 999;
        CHECK(view(i) == map(i));
    }

    Kokkos::View<double**,Kokkos::HostSpace> view2d("View", size,size);
    for(unsigned int j=0; j<size; ++j){
        for(unsigned int i=0; i<size; ++i){
            view2d(i,j) = i + j*size;
        }
    }

    SECTION("Subview column"){
        auto sub_view = Kokkos::subview(view2d, Kokkos::ALL(),6);
        auto map2 = KokkosToVec(sub_view);
        for(unsigned int i=0; i<size; ++i){
            CHECK(sub_view(i) == map2(i));
            sub_view(i) += 999;
            CHECK(sub_view(i) == map2(i));
        }
    }

    SECTION("Subview row"){
        auto sub_view = Kokkos::subview(view2d, 2, Kokkos::ALL());
        auto map2 = KokkosToVec(sub_view);
        for(unsigned int i=0; i<size; ++i){
            CHECK(sub_view(i) == map2(i));
            sub_view(i) += 999.;
            CHECK(sub_view(i) == map2(i));
        }
    }
}

TEST_CASE( "Testing copy Kokkos to Eigen Conversions in 1D", "[CopyKokkosToEigen1d]" ) {

    unsigned int size = 20;
    Kokkos::View<double*, Kokkos::HostSpace> view("View", size);
    for(unsigned int i=0; i<size; ++i)
        view(i) = i;

    Eigen::VectorXd map = CopyKokkosToVec<double>(view);
    for(unsigned int i=0; i<size; ++i){
        CHECK(view(i) == map(i));
        view(i) += 999;
        CHECK(view(i) != map(i));
    }

    Kokkos::View<double**,Kokkos::HostSpace> view2d("View", size,size);
    for(unsigned int j=0; j<size; ++j){
        for(unsigned int i=0; i<size; ++i){
            view2d(i,j) = i + j*size;
        }
    }
    
    SECTION("Subview column"){
        auto sub_view = Kokkos::subview(view2d, Kokkos::ALL(),6);
        auto map2 = CopyKokkosToVec(sub_view);
        for(unsigned int i=0; i<size; ++i){
            CHECK(sub_view(i) == map2(i));
            sub_view(i) += 999;
            CHECK(sub_view(i) != map2(i));
        }
    }

    SECTION("Subview row"){
        auto sub_view = Kokkos::subview(view2d, 2, Kokkos::ALL());
        auto map2 = CopyKokkosToVec(sub_view);
        for(unsigned int i=0; i<size; ++i){
            CHECK(sub_view(i) == map2(i));
            sub_view(i) += 999;
            CHECK(sub_view(i) != map2(i));
        }
    }
}


TEST_CASE( "Testing copy Kokkos to Eigen Conversions in 2D", "[CopyKokkosToEigen2d]" ) {

    
    unsigned int N = 8;
    unsigned int M = 10;

    SECTION("Row Major"){

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> view("View", N, M);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                view(i,j) = i + j*N;
            }
        }

        auto map = KokkosToMat<double>(view);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                CHECK(view(i,j) == map(i,j));
                view(i,j) += 999;
                CHECK(view(i,j) == map(i,j));
            }
        }
    }

    SECTION("Col Major"){

        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> view("View", N, M);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                view(i,j) = i + j*N;
            }
        }

        auto map = KokkosToMat<double>(view);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                CHECK(view(i,j) == map(i,j));
                view(i,j) += 999;
                CHECK(view(i,j) == map(i,j));
            }
        }
    }

    SECTION("Block of Row Major"){

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> view("View", N, M);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                view(i,j) = i + j*N;
            }
        }

        auto sub_view = Kokkos::subview(view, std::make_pair(2,4), std::make_pair(4,6) );

        auto map = KokkosToMat<double>(sub_view);
        for(unsigned int j=0; j<sub_view.extent(1); ++j){
            for(unsigned int i=0; i<sub_view.extent(0); ++i){
                CHECK(sub_view(i,j) == map(i,j));
                sub_view(i,j) += 999;
                CHECK(sub_view(i,j) == map(i,j));
            }
        }
    }


    SECTION("Block of Col Major"){

        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> view("View", N, M);
        for(unsigned int j=0; j<M; ++j){
            for(unsigned int i=0; i<N; ++i){
                view(i,j) = i + j*N;
            }
        }

        auto sub_view = Kokkos::subview(view, std::make_pair(2,4), std::make_pair(4,6) );

        auto map = KokkosToMat(sub_view);
        for(unsigned int j=0; j<sub_view.extent(1); ++j){
            for(unsigned int i=0; i<sub_view.extent(0); ++i){
                CHECK(sub_view(i,j) == map(i,j));
                sub_view(i,j) += 999;
                CHECK(sub_view(i,j) == map(i,j));
                sub_view(i,j) -= 999;
            }
        }


        int imult = 2;
        int jmult = 2;
        Kokkos::LayoutStride strides(2, imult, 3, jmult*view.extent(1));
        Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view2(view.data(), strides);

        auto map2 = KokkosToMat<double>(view2);

        for(unsigned int j=0; j<view2.extent(1); ++j){
            for(unsigned int i=0; i<view2.extent(0); ++i){
                CHECK(view2(i,j) == map2(i,j));
                CHECK(map2(i,j) == imult*i + jmult*j*M);
                view2(i,j) += 999;
                CHECK(view2(i,j) == map2(i,j));
            }
        }


        Eigen::MatrixXd mat = CopyKokkosToMat(view);
        for(unsigned int j=0; j<view.extent(1); ++j){
            for(unsigned int i=0; i<view.extent(0); ++i){
                CHECK(view(i,j) == mat(i,j));
                mat(i,j) += 999;
                CHECK(view(i,j) != mat(i,j));
            }
        }

    }

}
