#include <catch2/catch_all.hpp>

#include <Kokkos_Core.hpp>
#include "MParT/AffineMap.h"
#include "MParT/MapObjective.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Test KLMapObjective evaluation", "[KLMapObjective]") {
    unsigned int dim = 2;
    unsigned int seed = 170283;
    unsigned int N_samples = 200000;
    unsigned int N_testpts = N_samples/5;
    double map_scale = 2.5;
    double map_shift = 1.5;

    std::shared_ptr<GaussianSamplerDensity<Kokkos::HostSpace>> density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
    density->SetSeed(seed);
    StridedMatrix<double, Kokkos::HostSpace> reference_samples = density->Sample(N_samples);

    Kokkos::View<double**, Kokkos::HostSpace> A ("Map scale", dim, dim);
    Kokkos::View<double*, Kokkos::HostSpace> b ("Map shift", dim);;
    for(int i = 0; i < dim; i++) {
        b(i) = map_shift;
        for(int j = 0; j < dim; j++) {
            A(i,j) = map_scale*((double) i == j);
        }
    }
    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A,b);
    KLObjective<Kokkos::HostSpace> objective {reference_samples, density};
    std::vector<double> kl_ests{};
    unsigned int init_pts = 10;
    double kl_est = 0.;
    while(init_pts <= N_samples) {
        auto samps = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, init_pts));
        kl_est = objective.ObjectiveImpl(samps, map);
        kl_ests.push_back(kl_est);
        std::cout << "N = " << init_pts << ", kl_est = " << kl_est << "\n";
        init_pts *= 2;
    }

    double ms_const = map_scale*map_scale;
    double kl_exact = -std::log(ms_const)-1 + ms_const + map_shift*map_shift*ms_const;
    kl_exact /= 2.;
    std::cout << "kl_est = " << kl_est << ", kl_exact = " << kl_exact << std::endl;
}