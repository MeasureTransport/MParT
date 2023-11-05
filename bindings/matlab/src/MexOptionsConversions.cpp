#include "MexOptionsConversions.h"
#include <iostream>

using namespace mpart;

MapOptions  mpart::binding::MapOptionsFromMatlab(std::string basisType, std::string posFuncType,
                                        std::string quadType, double quadAbsTol,
                                        double quadRelTol, unsigned int quadMaxSub,
                                        unsigned int quadMinSub,unsigned int quadPts,
                                        bool contDeriv, double basisLB, double basisUB, bool basisNorm, double nugget)
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

    opts.quadAbsTol = quadAbsTol;
    opts.quadRelTol = quadRelTol;
    opts.quadMaxSub = quadMaxSub;
    opts.quadMinSub = quadMinSub;
    opts.quadPts = quadPts;
    opts.contDeriv = contDeriv;
    opts.basisLB = basisLB;
    opts.basisUB = basisUB;
    opts.basisNorm = basisNorm;
    opts.nugget = nugget;
    return opts;
}

void mpart::binding::MapOptionsToMatlab(MapOptions opts, mexplus::OutputArguments &output, int start)
{
    int i = start; // Alias
    switch(opts.basisType) {
        case BasisTypes::ProbabilistHermite: output.set(i,"ProbabilistHermite");break;
        case BasisTypes::PhysicistHermite: output.set(i,"PhysicistHermite");break;
        case BasisTypes::HermiteFunctions: output.set(i,"HermiteFunctions");break;
        default: output.set(i,"");break;
    }
    switch(opts.posFuncType) {
        case PosFuncTypes::Exp: output.set(i+1,"Exp");break;
        case PosFuncTypes::SoftPlus: output.set(i+1,"SoftPlus");break;
        default: output.set(i+1,"");break;
    }
    switch(opts.quadType) {
        case QuadTypes::ClenshawCurtis: output.set(i+2,"ClenshawCurtis"); break;
        case QuadTypes::AdaptiveSimpson: output.set(i+2,"AdaptiveSimpson");break;
        case QuadTypes::AdaptiveClenshawCurtis: output.set(i+2,"AdaptiveClenshawCurtis");break;
        default: output.set(i+2,"");break;
    }

    output.set(i+3,opts.quadAbsTol);
    output.set(i+4,opts.quadRelTol);
    output.set(i+5,opts.quadMaxSub);
    output.set(i+6,opts.quadMinSub);
    output.set(i+7,opts.quadPts);
    output.set(i+8,opts.contDeriv);
    output.set(i+9,opts.basisLB);
    output.set(i+10,opts.basisUB);
    output.set(i+11,opts.basisNorm);
    output.set(i+12,opts.nugget);
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
    MultiIndex& maxDegrees = *mexplus::Session<MultiIndex>::get(input.get(start+23));
    return ATMOptionsFromMatlab(input.get<std::string>(start + 0), input.get<std::string>(start + 1),
                                input.get<std::string>(start + 2), input.get<double>(start + 3),
                                input.get<double>(start + 4), input.get<unsigned int>(start + 5),
                                input.get<unsigned int>(start + 6),input.get<unsigned int>(start + 7),
                                input.get<bool>(start + 8), input.get<double>(start + 9),
                                input.get<double>(start + 10), input.get<bool>(start + 11),
                                input.get<std::string>(start + 12), input.get<double>(start + 13),
                                input.get<double>(start + 14), input.get<double>(start + 15),
                                input.get<double>(start + 16), input.get<double>(start + 17),
                                input.get<int>(start + 18), input.get<double>(start + 19),
                                input.get<int>(start + 20), input.get<unsigned int>(start + 21),
                                input.get<unsigned int>(start + 22), maxDegrees);
}

ATMOptions  mpart::binding::ATMOptionsFromMatlab(std::string basisType, std::string posFuncType,
                                        std::string quadType, double quadAbsTol,
                                        double quadRelTol, unsigned int quadMaxSub,
                                        unsigned int quadMinSub,unsigned int quadPts,
                                        bool contDeriv, double basisLB, double basisUB, bool basisNorm,
                                        std::string opt_alg, double opt_stopval,
                                        double opt_ftol_rel, double opt_ftol_abs,
                                        double opt_xtol_rel, double opt_xtol_abs,
                                        int opt_maxeval, double opt_maxtime, int verbose,
                                        unsigned int maxPatience, unsigned int maxSize, MultiIndex& maxDegrees)
{
    ATMOptions opts;

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

    opts.quadAbsTol = quadAbsTol;
    opts.quadRelTol = quadRelTol;
    opts.quadMaxSub = quadMaxSub;
    opts.quadMinSub = quadMinSub;
    opts.quadPts = quadPts;
    opts.contDeriv = contDeriv;
    opts.basisLB = basisLB;
    opts.basisUB = basisUB;
    opts.basisNorm = basisNorm;
    opts.opt_alg = opt_alg;
    opts.opt_stopval = opt_stopval;
    opts.opt_ftol_rel = opt_ftol_rel;
    opts.opt_ftol_abs = opt_ftol_abs;
    opts.opt_xtol_rel = opt_xtol_rel;
    opts.opt_xtol_abs = opt_xtol_abs;
    opts.opt_maxeval = opt_maxeval;
    opts.opt_maxtime = opt_maxtime;
    opts.verbose = verbose;
    opts.maxPatience = maxPatience;
    opts.maxSize = maxSize;
    opts.maxDegrees = maxDegrees;

    return opts;
}
#endif // defined(MPART_HAS_NLOPT)