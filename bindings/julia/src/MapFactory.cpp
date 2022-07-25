#include "MParT/MapFactory.h"
#include "CommonJuliaUtilities.h"

void mpart::binding::MapFactoryWrapper(jlcxx::Module &mod) {
    // CreateComponent
    mod.method("CreateComponent", &MapFactory::CreateComponent<Kokkos::HostSpace>);

    // CreateTriangular
    mod.method("CreateTriangular", &MapFactory::CreateTriangular<Kokkos::HostSpace>);
}