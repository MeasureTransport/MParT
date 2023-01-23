#include <catch2/catch_all.hpp>

#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"
#include "MParT/AdaptiveTransportMap.h"

using namespace mpart;
using namespace Catch;

#include <fstream>

void SaveMatrix(std::string fname, StridedMatrix<double, Kokkos::HostSpace> mat, std::string path="/home/dannys4/misc/mpart_atm/") {
    path = path + fname;
    std::ofstream file {path, std::ios::out};
    for(int i = 0; i < mat.extent(0); i++) {
        for(int j = 0; j < mat.extent(1); j++) {
            file << mat(i,j);
            if(j < mat.extent(1) - 1) file << ",";
        }
        file << "\n";
    }
}

void ATM() {
    unsigned int seed = 20;
    unsigned int dim = 2;
    unsigned int numPts = 20000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(2);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, Kokkos::HostSpace> targetSamps("targetSamps", 2, numPts);
    for(int i = 0; i < numPts; i++){
        targetSamps(0,i) = samples(0,i);
        targetSamps(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
    };
    std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(1,0), MultiIndexSet::CreateTotalOrder(2,0)};
    ATMOptions opts;
    opts.opt_alg = "LD_SLSQP";
    opts.basisType = BasisTypes::ProbabilistHermite;
    opts.maxSize = 8;
    opts.maxPatience = 3;
    opts.opt_xtol_rel = 1e-16;
    opts.opt_xtol_abs = 1e-16;
    opts.verbose = true;
    StridedMatrix<double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    StridedMatrix<double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    KLObjective<Kokkos::HostSpace> objective {trainSamps,testSamps,sampler};
    auto atm = AdaptiveTransportMap<Kokkos::HostSpace>(mset0, objective, opts);
    Kokkos::View<double**, Kokkos::HostSpace> null_prefix ("Null prefix", 0, numPts);
    auto forward_samps = atm->Evaluate(targetSamps);
    auto inv_samps = atm->Inverse(null_prefix, targetSamps);
}

/*void TestNLL() {
    unsigned int dim = 2;
    unsigned int numPts = 3;
    Kokkos::View<double**, Kokkos::HostSpace> x("x", dim, numPts);
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < numPts; j++) {
            x(i,j) = dim*j + i + 1;
        }
    }

    ATMOptions opts;
    opts.basisType = BasisTypes::ProbabilistHermite;
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, 1);
    auto comp = MapFactory::CreateComponent(mset.Fix(true), opts);
    Kokkos::View<double*, Kokkos::HostSpace> coeffs("coeffs", comp->numCoeffs);
    Kokkos::parallel_for("init coeffs", comp->numCoeffs, KOKKOS_LAMBDA(const unsigned int i) {
        coeffs(i) = 1;
    });
    StridedMatrix<double, Kokkos::HostSpace> xStrided = x;
    ATMObjective obj {xStrided, xStrided, comp, opts};
    std::vector<double> grad(comp->numCoeffs);
    double nll = obj(comp->numCoeffs, coeffs.data(), grad.data());
    std::cout << "nll = " << nll << "\n";
    for(int i = 0; i < comp->numCoeffs; i++) {
        std::cout << "grad[ " << i << "] = " << grad[i] << "\n";
    }
    std::cout << std::endl;
}*/