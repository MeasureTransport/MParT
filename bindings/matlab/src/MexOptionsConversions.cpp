#include "MexOptionsConversions.h"
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

void mpart::binding::MapOptionsToMatlab(MapOptions opts, OutputArguments &output, int start = 0)
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
}