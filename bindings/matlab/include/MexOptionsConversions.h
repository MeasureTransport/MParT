#ifndef MPART_MEXOPTIONSCONVERSIONS_H
#define MPART_MEXOPTIONSCONVERSIONS_H

#include <iostream>
#include <mexplus.h>
#include "MParT/MapOptions.h"

namespace mpart{
namespace binding{

    /** Converts a real-valued matlab vector to a Kokkos::View.  The memory in matlab vector is not copied for performance
        reasons.  However, this means that the user is responsible for ensuring the vector is not freed before the view.
    */
    MapOptions  MapOptionsFromMatlab(std::string basisType, std::string posFuncType, 
                                        std::string quadType, double quadAbsTol,
                                        double quadRelTol, unsigned int quadMaxSub, 
                                        unsigned int quadMinSub,unsigned int quadPts, 
                                        bool contDeriv, double basisLB, double basisUB, bool basisNorm);

    void MapOptionsToMatlab(MapOptions opts, OutputArguments &output);
}
}


#endif 