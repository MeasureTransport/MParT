#include "MexOptionsConversions.h"
#include <iostream>

using namespace mpart;

#define MPART_MEX_MAPOPTIONS_ARGCOUNT 16
#define MPART_MEX_TRAINOPTIONS_ARGCOUNT 9

MapOptions MapOptionsFromMatlabArgs(
    std::string basisType, std::string sigmoidType,
    std::string edgeType, std::string posFuncType,
    std::string quadType, double quadAbsTol,
    double quadRelTol, unsigned int quadMaxSub,
    unsigned int quadMinSub, double edgeShape,
    unsigned int quadPts, bool contDeriv, double basisLB,
    double basisUB, bool basisNorm, double nugget)
{
    MapOptions opts;

    if (basisType == "ProbabilistHermite") {
    opts.basisType    = BasisTypes::ProbabilistHermite;
    } else if (basisType == "PhysicistHermite") {
    opts.basisType    = BasisTypes::PhysicistHermite;
    } else if (basisType == "HermiteFunctions") {
    opts.basisType    = BasisTypes::HermiteFunctions;
    } else {
    std::cout << "Unknown basisType, value is set to default" <<std::endl;
    }

    if (posFuncType == "Exp") {
    opts.posFuncType    = PosFuncTypes::Exp;
    } else if (posFuncType == "SoftPlus") {
    opts.posFuncType    = PosFuncTypes::SoftPlus;
    } else {
    std::cout << "Unknown posFuncType type, value is set to default" <<std::endl;
    }

    if (quadType == "ClenshawCurtis") {
    opts.quadType    = QuadTypes::ClenshawCurtis;
    } else if (quadType == "AdaptiveSimpson") {
    opts.quadType    = QuadTypes::AdaptiveSimpson;
    } else if (quadType == "AdaptiveClenshawCurtis") {
    opts.quadType    = QuadTypes::AdaptiveClenshawCurtis;
    } else {
    std::cout << "Unknown quadType, value is set to default" <<std::endl;
    }

    if (sigmoidType == "Logistic") {
    opts.sigmoidType    = SigmoidTypes::Logistic;
    } else {
    std::cout << "Unknown sigmoidType, value is set to default" <<std::endl;
    }

    if (edgeType == "SoftPlus") {
    opts.edgeType    = EdgeTypes::SoftPlus;
    } else {
    std::cout << "Unknown edgeType, value is set to default" <<std::endl;
    }

    opts.quadAbsTol = quadAbsTol;
    opts.quadRelTol = quadRelTol;
    opts.quadMaxSub = quadMaxSub;
    opts.quadMinSub = quadMinSub;
    opts.edgeShape = edgeShape;
    opts.quadPts = quadPts;
    opts.contDeriv = contDeriv;
    opts.basisLB = basisLB;
    opts.basisUB = basisUB;
    opts.basisNorm = basisNorm;
    opts.nugget = nugget;
    return opts;
}


MapOptions mpart::binding::MapOptionsFromMatlab(mexplus::InputArguments &input, int start) {
    return MapOptionsFromMatlabArgs(
        input.get<std::string>(start + 0), input.get<std::string>(start + 1),
        input.get<std::string>(start + 2), input.get<std::string>(start + 3),
        input.get<std::string>(start + 4), input.get<double>(start + 5),
        input.get<double>(start + 6), input.get<unsigned int>(start + 7),
        input.get<unsigned int>(start + 8), input.get<double>(start + 9),
        input.get<unsigned int>(start + 10), input.get<bool>(start + 11),
        input.get<double>(start + 12), input.get<double>(start + 13),
        input.get<bool>(start + 14), input.get<double>(start + 15)
    );
}

void mpart::binding::MapOptionsToMatlab(MapOptions opts, mexplus::OutputArguments &output, int start)
{
    int i = start; // Alias
    output.set(i++, MapOptions::btypes[static_cast<unsigned int>(opts.basisType)]); // basisType
    output.set(i++, MapOptions::stypes[static_cast<unsigned int>(opts.sigmoidType)]); // sigmoidType
    output.set(i++, MapOptions::etypes[static_cast<unsigned int>(opts.edgeType)]); // edgeType
    output.set(i++, MapOptions::pftypes[static_cast<unsigned int>(opts.posFuncType)]); // posFuncType
    output.set(i++, MapOptions::qtypes[static_cast<unsigned int>(opts.quadType)]); // quadType
    
    constexpr int numScalars = 10;
    // bools describe which are integers
    const std::pair<double,bool> optsScalars[numScalars] = {
        {double(opts.quadAbsTol),0},
        {double(opts.quadRelTol),0},
        {double(opts.quadMaxSub),1},
        {double(opts.quadMinSub),1},
        {double(opts.edgeShape),0},
        {double(opts.quadPts),1},
        {double(opts.basisLB),0},
        {double(opts.basisUB),0},
        {double(opts.nugget),0}
    };
    for(int j = 0; j < numScalars; j++) {
        std::pair<double, bool> p = optsScalars[j];
        if(p.second) {
            output.set(i+j,static_cast<unsigned int>(p.first));
        } else {
            output.set(i+j,p.first);
        }
    }
}

#if defined(MPART_HAS_NLOPT)

TrainOptions mpart::binding::TrainOptionsFromMatlab(mexplus::InputArguments &input, unsigned int start) {
    TrainOptions opts;
    opts.opt_alg = input.get<std::string>(start + 0);
    opts.opt_stopval = input.get<double>(start + 1);
    opts.opt_ftol_rel = input.get<double>(start + 2);
    opts.opt_ftol_abs = input.get<double>(start + 3);
    opts.opt_xtol_rel = input.get<double>(start + 4);
    opts.opt_xtol_abs = input.get<double>(start + 5);
    opts.opt_maxeval = input.get<int>(start + 6);
    opts.opt_maxtime = input.get<double>(start + 7);
    opts.verbose = input.get<int>(start + 8);
    return opts;
}

ATMOptions mpart::binding::ATMOptionsFromMatlab(mexplus::InputArguments &input, unsigned int start) {
    MapOptions opts = MapOptionsFromMatlab(input, start);
    TrainOptions trainOpts = TrainOptionsFromMatlab(input, start + MPART_MEX_MAPOPTIONS_ARGCOUNT);
    unsigned int atmStart = start + MPART_MEX_MAPOPTIONS_ARGCOUNT + MPART_MEX_TRAINOPTIONS_ARGCOUNT;
    MultiIndex& maxDegrees = *mexplus::Session<MultiIndex>::get(input.get(atmStart));
    unsigned int maxPatience = input.get<unsigned int>(atmStart + 1);
    unsigned int maxSize = input.get<unsigned int>(atmStart + 2);
    return ATMOptions(opts, trainOpts, maxDegrees, maxPatience, maxSize);
}
#endif // defined(MPART_HAS_NLOPT)