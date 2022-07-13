#include "MParT/Initialization.h"

using namespace mpart;


void mpart::Finalize()
{
    std::cout << "Before kokkos::finalize..." << std::endl;
    Kokkos::finalize();
    std::cout << "After kokkos::finalize..." << std::endl;
}

InitializeStatus& mpart::GetInitializeStatusObject()
{
    static InitializeStatus status;

    return status;
}