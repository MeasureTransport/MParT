#include "MexMapOptionsConversions.h"
#include <iostream>

using namespace mpart;

MapOptions  mpart::binding::MapOptionsFromMatlab(std::string basisType, std::string posFuncType, 
                                        std::string quadType, double quadAbsTol,
                                        double quadRelTol, unsigned int quadMaxSub, 
                                        unsigned int quadMinSub,unsigned int quadPts, 
                                        bool contDeriv, double basisLB, double basisUB, bool basisNorm)
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
    return opts;
}
