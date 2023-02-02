#ifndef MPART_MEXOPTIONSCONVERSIONS_H
#define MPART_MEXOPTIONSCONVERSIONS_H

#include <iostream>
#include <mexplus.h>
#include "MParT/MapOptions.h"
#if defined(MPART_HAS_NLOPT)
#include "MParT/AdaptiveTransportMap.h"
#endif // defined(MPART_HAS_NLOPT)

namespace mpart{
namespace binding{

    MapOptions  MapOptionsFromMatlab(std::string basisType, std::string posFuncType,
                                     std::string quadType, double quadAbsTol,
                                     double quadRelTol, unsigned int quadMaxSub,
                                     unsigned int quadMinSub,unsigned int quadPts,
                                     bool contDeriv, double basisLB, double basisUB, bool basisNorm);

    void MapOptionsToMatlab(MapOptions opts, mexplus::OutputArguments &output, int start = 0);
#if defined(MPART_HAS_NLOPT)
    ATMOptions ATMOptionsFromMatlab(InputArguments input, unsigned int start);
    ATMOptions ATMOptionsFromMatlab(std::string basisType, std::string posFuncType,
                                    std::string quadType, double quadAbsTol,
                                    double quadRelTol, unsigned int quadMaxSub,
                                    unsigned int quadMinSub,unsigned int quadPts,
                                    bool contDeriv, double basisLB, double basisUB, bool basisNorm,
                                    std::string opt_alg, double opt_stopval,
                                    double opt_ftol_rel, double opt_ftol_abs,
                                    double opt_xtol_rel, double opt_xtol_abs,
                                    int opt_maxeval, double opt_maxtime, int verbose,
                                    unsigned int maxPatience, unsigned int maxSize, MultiIndex maxDegrees);
#endif // defined(MPART_HAS_NLOPT)
}
}


#endif