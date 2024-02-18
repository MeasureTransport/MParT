#ifndef MPART_MEXOPTIONSCONVERSIONS_H
#define MPART_MEXOPTIONSCONVERSIONS_H

#include <iostream>
#include <mexplus.h>
#include "MParT/MapOptions.h"
#if defined(MPART_HAS_NLOPT)
#include "MParT/TrainMapAdaptive.h"
#endif // defined(MPART_HAS_NLOPT)

namespace mpart{
namespace binding{

    const unsigned int MPART_MEX_MAPOPTIONS_ARGCOUNT = 16;
    const unsigned int MPART_MEX_TRAINOPTIONS_ARGCOUNT = 9;

    MapOptions MapOptionsFromMatlab(mexplus::InputArguments &input, int start);

    void MapOptionsToMatlab(MapOptions opts, mexplus::OutputArguments &output, int start = 0);
#if defined(MPART_HAS_NLOPT)
    TrainOptions TrainOptionsFromMatlab(mexplus::InputArguments &input, unsigned int start);
    ATMOptions ATMOptionsFromMatlab(mexplus::InputArguments &input, unsigned int start);
#endif // defined(MPART_HAS_NLOPT)
}
}


#endif