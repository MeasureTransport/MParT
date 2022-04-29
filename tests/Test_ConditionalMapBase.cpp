#include <catch2/catch_all.hpp>

#include "MParT/ConditionalMapBase.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing coefficient functions of conditional map base class", "[ConditionalMapBase]" ) {

    class MyIdentityMap : public ConditionalMapBase{
    public:
        MyIdentityMap(unsigned int dim) : ConditionalMapBase(dim,dim){};
        virtual ~MyIdentityMap() = default;
        virtual Kokkos::View<double**, Kokkos::HostSpace> Evaluate(Kokkos::View<double**, Kokkos::HostSpace> const& pts) override{return pts;};
        
        virtual Kokkos::View<double**, Kokkos::HostSpace> Inverse(Kokkos::View<double**, Kokkos::HostSpace> const& x1, 
                                                                  Kokkos::View<double**, Kokkos::HostSpace> const& r) override{return r;};
    };


    MyIdentityMap map(4);
    CHECK(map.inputDim == 4);
    CHECK(map.outputDim == 4);

    unsigned int numCoeffs = 10;
    Kokkos::View<double*, Kokkos::HostSpace> coeffs("New Coeffs", numCoeffs);
    for(unsigned int i=0; i<numCoeffs; ++i)
        coeffs(i) = i;
    
    map.Coeffs() = coeffs;
    CHECK(map.Coeffs().extent(0) == numCoeffs);

    for(unsigned int i=0; i<numCoeffs; ++i)
        CHECK(map.Coeffs()(i) == coeffs(i));

    coeffs(0) = 100;
    for(unsigned int i=0; i<numCoeffs; ++i)
        CHECK(map.Coeffs()(i) == coeffs(i));


    // Now check using a slice of the coefficients
    unsigned int start = 2;
    unsigned int end = 4;
    auto coeff_slice = Kokkos::subview(coeffs, std::make_pair(start, end));

    map.Coeffs() = coeff_slice;
    CHECK(coeffs.extent(0) == numCoeffs);
    CHECK(map.Coeffs().extent(0)==(end-start));

    for(unsigned int i=0; i<end-start; ++i)
        CHECK(map.Coeffs()(i)==coeffs(i+start));

    coeffs(start) = 1024;
    for(unsigned int i=0; i<end-start; ++i)
        CHECK(map.Coeffs()(i)==coeffs(i+start));

}