#include "CommonJuliaUtilities.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"

#include <Kokkos_Core.hpp>
#include "JlArrayConversions.h"

using namespace mpart;
using namespace mpart::binding;

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for TriangularMap
    template<> struct SuperType<mpart::KLObjective<Kokkos::HostSpace>> {typedef mpart::MapObjective<Kokkos::HostSpace> type;};
}


void mpart::binding::MapObjectiveWrapper(jlcxx::Module &mod) {
    using MemorySpace = Kokkos::HostSpace;
    std::string tName = "GaussianKLObjective";

    mod.add_type<MapObjective<MemorySpace>>("MapObjective")
        .method("TrainError", &MapObjective<MemorySpace>::TrainError)
        .method("TestError", &MapObjective<MemorySpace>::TestError)
    ;

    mod.add_type<KLObjective<MemorySpace>>(tName,jlcxx::julia_base_type<MapObjective<MemorySpace>>());
    mod.method("Create"+tName, [](jlcxx::ArrayRef<double,2> train) {
        StridedMatrix<const double, MemorySpace> trainView = JuliaToKokkos(train);
        Kokkos::View<double**,MemorySpace> storeTrain ("Training data", trainView.extent(0), trainView.extent(1));
        Kokkos::deep_copy(storeTrain, trainView);
        trainView = storeTrain;
        return ObjectiveFactory::CreateGaussianKLObjective(trainView);
    });
    mod.method("Create"+tName, [](jlcxx::ArrayRef<double,2> train, jlcxx::ArrayRef<double,2> test) {
        StridedMatrix<const double, MemorySpace> trainView = JuliaToKokkos(train);
        StridedMatrix<const double, MemorySpace> testView = JuliaToKokkos(test);
        Kokkos::View<double**,MemorySpace> storeTrain ("Training data", trainView.extent(0), trainView.extent(1));
        Kokkos::View<double**,MemorySpace> storeTest ("Testing data", testView.extent(0), testView.extent(1));
        Kokkos::deep_copy(storeTrain, trainView);
        Kokkos::deep_copy(storeTest, testView);
        trainView = storeTrain;
        testView = storeTest;
        return ObjectiveFactory::CreateGaussianKLObjective(trainView, testView);
    });
}

