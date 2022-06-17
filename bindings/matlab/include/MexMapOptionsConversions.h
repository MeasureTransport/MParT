#ifndef MPART_MEXMAPOPTIONSCONVERSIONS_H
#define MPART_MEXMAPOPTIONSCONVERSIONS_H

#include <iostream>
#include "MParT/MapOptions.h"

namespace mpart{

    /** Converts a real-valued matlab vector to a Kokkos::View.  The memory in matlab vector is not copied for performance
        reasons.  However, this means that the user is responsible for ensuring the vector is not freed before the view.
    */
    MapOptions  MapOptionsFromMatlab(std::string basisType, std::string posFuncType, 
                                        std::string quadType, double quadAbsTol,
                                        double quadRelTol, unsigned int quadMaxSub,
                                        unsigned int quadPts);
}


#endif 