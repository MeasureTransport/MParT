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

        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> x2(rows,cols);
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