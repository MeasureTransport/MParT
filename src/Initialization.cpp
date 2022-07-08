#include "MParT/Initialization.h"

using namespace mpart;


void mpart::Finalize()
{
    Kokkos::finalize();
}

InitializeStatus& mpart::GetInitializeStatusObject()
{
    static InitializeStatus status;

    return status;
}