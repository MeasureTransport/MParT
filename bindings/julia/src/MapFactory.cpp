#include "MParT/MapFactory.h"
#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"
#include "jlcxx/stl.hpp"

void mpart::binding::MapFactoryWrapper(jlcxx::Module &mod) {
    // CreateComponent
    mod.method("CreateComponent", &MapFactory::CreateComponent<Kokkos::HostSpace>);

    // CreateTriangular
    mod.method("CreateTriangular", &MapFactory::CreateTriangular<Kokkos::HostSpace>);

    // CreateSigmoidComponent
    mod.method("CreateSigmoidComponent", [](unsigned int inDim, unsigned int totalOrder, jlcxx::ArrayRef<double,1> centers, MapOptions opts){
        StridedVector<const double, Kokkos::HostSpace> centersVec = JuliaToKokkos(centers);
        return MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(inDim, totalOrder, centersVec, opts);
    });

    // CreateSigmoidTriangular
    mod.method("CreateSigmoidTriangular", [](unsigned int inDim, unsigned int outDim, unsigned int totalOrder, jlcxx::ArrayRef<double,2> centers, MapOptions opts){
        std::vector<StridedVector<const double, Kokkos::HostSpace>> centersVecs;
        StridedMatrix<double, Kokkos::HostSpace> centersMat = JuliaToKokkos(centers);
        for(unsigned int i = 0; i < size(centers, 1); i++){
            Kokkos::View<double*, Kokkos::HostSpace> centersVec ("Centers i", size(centers, 0));
            StridedVector<double, Kokkos::HostSpace> center_i = Kokkos::subview(centersMat, Kokkos::ALL(), i);
            Kokkos::deep_copy(centersVec, center_i);
            centersVecs.push_back(centersVec);
        }
        return MapFactory::CreateSigmoidTriangular<Kokkos::HostSpace>(inDim, outDim, totalOrder, centersVecs, opts);
    });
}