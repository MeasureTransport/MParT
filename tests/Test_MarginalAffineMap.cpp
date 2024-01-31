#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include "MParT/MarginalAffineMap.h"
#include "MParT/IdentityMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;

using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Test MarginalAffineMap", "[MarginalAffineMap]") {
    SECTION("IdentityMap, square") {
        unsigned int dim = 2;
        double scale_base = 2.3;
        double shift_base = 1.2;
        auto id = std::make_shared<IdentityMap<MemorySpace>>(dim,dim);
        Kokkos::View<double*, MemorySpace> scale ("Map scale", dim);
        Kokkos::View<double*, MemorySpace> shift ("Map shift", dim);
        Kokkos::parallel_for("Fill scale and shift", dim, KOKKOS_LAMBDA(const unsigned int i) {
            scale(i) = scale_base + i;
            shift(i) = shift_base + i;
        });
        auto map = std::make_shared<MarginalAffineMap<MemorySpace>>(scale, shift, id);
        int N_pts = 10;
        Kokkos::View<double**, MemorySpace> pts("pts", dim, N_pts);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {dim, N_pts});
        Kokkos::parallel_for("Fill pts", policy, KOKKOS_LAMBDA(const unsigned int& i, const unsigned int& j){
            pts(i,j) = i+j;
        });
        // Evaluate
        StridedMatrix<double, MemorySpace> output = map->Evaluate(pts);
        // Check
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < N_pts; j++) {
                REQUIRE_THAT(output(i,j), WithinRel(scale(i)*pts(i,j) + shift(i), 1e-14));
            }
        }
    }
}