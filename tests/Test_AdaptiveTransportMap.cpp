#include <catch2/catch_all.hpp>

#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"
#include "MParT/AdaptiveTransportMap.h"

using namespace mpart;
using namespace Catch;

void ATM() {
    unsigned int seed = 155829;
    unsigned int dim = 2;
    unsigned int numPts = 20000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(2);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, Kokkos::HostSpace> targetSamps("targetSamps", 2, numPts);
    Kokkos::parallel_for("Banana", numPts, KOKKOS_LAMBDA(const unsigned int i) {
        targetSamps(0,i) = samples(0,i);
        targetSamps(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
    });
    std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(1,0), MultiIndexSet::CreateTotalOrder(2,0)};
    ATMOptions opts;
    opts.basisType = BasisTypes::ProbabilistHermite;
    opts.maxSize = 16;
    opts.maxPatience = 5;
    opts.verbose = true;
    StridedMatrix<double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    StridedMatrix<double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    KLObjective<Kokkos::HostSpace> objective {trainSamps,testSamps,sampler};
    auto atm = AdaptiveTransportMap<Kokkos::HostSpace>(mset0, objective, opts);
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