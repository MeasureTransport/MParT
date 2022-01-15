#include <catch2/catch_test_macros.hpp>

#include "MParT/MultiIndices/MultiIndex.h"

using namespace mpart;

TEST_CASE( "Testing MultiIndex", "[MultiIndex]" ) {

    const unsigned int length = 4;

    MultiIndex multi;
    REQUIRE( multi.Length()==0 );
    
    // Use the constructor to create a multiindex of all zeros
    multi = MultiIndex(length);
    REQUIRE( multi.Length()==length );
    REQUIRE( multi.NumNz() == 0);
    REQUIRE( multi.Get(1) == 0);
    REQUIRE( multi.Max() == 0);
    REQUIRE( multi.Sum() == 0);

    std::vector<unsigned int> dense;
    dense = multi.Vector();
    for(auto& val : dense)
        REQUIRE( val==0 );

    std::string denseStr;
    denseStr = multi.String();
    REQUIRE( denseStr == "0 0 0 0");

    // Set one of the values 
    multi.Set(2,1);

    REQUIRE( multi.Length()==length );
    REQUIRE( multi.NumNz() == 1);
    REQUIRE( multi.Get(1) == 0);
    REQUIRE( multi.Get(2) == 1);
    REQUIRE( multi.Max() == 1);
    REQUIRE( multi.Sum() == 1);

    dense = multi.Vector();
    for(int i=0; i<dense.size(); ++i){
        if(i!=2){
            REQUIRE( dense[i] == 0);
        }else{
            REQUIRE( dense[i] == 1);
        }
    }

    denseStr = multi.String();
    REQUIRE( denseStr == "0 0 1 0");


    // Set another new value 
    multi.Set(1,2);

    REQUIRE( multi.Length()==length );
    REQUIRE( multi.NumNz() == 2);
    REQUIRE( multi.Get(1) == 2);
    REQUIRE( multi.Get(2) == 1);
    REQUIRE( multi.Max() == 2);
    REQUIRE( multi.Sum() == 3);

    denseStr = multi.String();
    REQUIRE( denseStr == "0 2 1 0");


    // Override the index 2
    multi.Set(2,3);

    REQUIRE( multi.Length()==length );
    REQUIRE( multi.NumNz() == 2);
    REQUIRE( multi.Get(1) == 2);
    REQUIRE( multi.Get(2) == 3);
    REQUIRE( multi.Max() == 3);
    REQUIRE( multi.Sum() == 5);

    denseStr = multi.String();
    REQUIRE( denseStr == "0 2 3 0");


    dense.resize(4,0);
    dense[0] = 1;
    dense[1] = 2;
    dense[2] = 1;

    multi = MultiIndex(dense);
    REQUIRE(multi.NumNz()==3);
    REQUIRE(multi.Get(0)==1);
    REQUIRE(multi.Get(1)==2);
    REQUIRE(multi.Get(2)==1);
    REQUIRE(multi.Get(3)==0);
}

TEST_CASE( "Testing MultiIndex ordering", "[MultiIndexOrder]" ) {

    MultiIndex a({0,1,1,2});
    MultiIndex b({1,0,0,0});

    REQUIRE( b<a );
    REQUIRE( a>b );
    REQUIRE( b<=a );
    REQUIRE( a>=b );
    REQUIRE( b!=a );

    b = MultiIndex({0,1,1,2});
    REQUIRE( b==a );
    REQUIRE( b<=a );
    REQUIRE( a>=b );

    b = MultiIndex({1,2,1,0});
    REQUIRE( a<b );
    REQUIRE( a<=b );
    REQUIRE( b>a );
    REQUIRE( b>=a );
    REQUIRE( a!=b );
}