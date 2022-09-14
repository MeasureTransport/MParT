#include "MParT/Initialization.h"

using namespace mpart;


void mpart::Finalize()
{
#if defined(MPART_ENABLE_GPU)
    magma_finalize();
#endif 

    Kokkos::finalize();
}

InitializeStatus& mpart::GetInitializeStatusObject()
{
    static InitializeStatus status;

    return status;
}