#include "MParT/Distributions/DensityBase.h"
#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/Distribution.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/Distributions/PullbackDensity.h"
#include "MParT/Distributions/PullbackSampler.h"
#include "MParT/Distributions/PushforwardDensity.h"
#include "MParT/Distributions/PushforwardSampler.h"
#include "MParT/Distributions/TransportDistributionFactory.h"


#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

using MemorySpace = Kokkos::HostSpace;
using namespace mpart;

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for DensityBase and SampleGenerator
    template<> struct SuperType<GaussianSamplerDensity<MemorySpace>> {typedef DensityBase<MemorySpace> type;};
    template<> struct SuperType<PullbackDensity<MemorySpace>> {typedef DensityBase<MemorySpace> type;};
    template<> struct SuperType<PushforwardDensity<MemorySpace>> {typedef DensityBase<MemorySpace> type;};

    template<> struct SuperType<PullbackSampler<MemorySpace>> {typedef SampleGenerator<MemorySpace> type;};
    template<> struct SuperType<PushforwardSampler<MemorySpace>> {typedef SampleGenerator<MemorySpace> type;};
}

void mpart::binding::DistributionsWrapper(jlcxx::Module &mod) {
    mod.add_type<DensityBase<MemorySpace>>("DensityBase")
    .method("LogDensity", [](DensityBase<MemorySpace> &density, jlcxx::ArrayRef<double,2> pts){
        unsigned int numPts = size(pts,1);
        jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
        density.LogDensityImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
    })
    .method("LogDensityInputGrad", [](DensityBase<MemorySpace> &density, jlcxx::ArrayRef<double,2> pts){
        unsigned int numPts = size(pts,1);
        unsigned int numInputs = density.Dim();
        jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numInputs, numPts);
        density.LogDensityInputGradImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
        return output;
    })
    ;
    mod.add_type<SampleGenerator<MemorySpace>>("SampleGenerator");
    mod.add_type<Distribution<MemorySpace>>("Distribution")
    .method("LogDensity", [](Distribution<MemorySpace> &density, jlcxx::ArrayRef<double,2> pts){
        unsigned int numPts = size(pts,1);
        jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
        density.LogDensityImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
        return output;
    })
    .method("LogDensityInputGrad", [](Distribution<MemorySpace> &density, jlcxx::ArrayRef<double,2> pts){
        unsigned int numPts = size(pts,1);
        unsigned int numInputs = density.Dim();
        jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numInputs, numPts);
        density.LogDensityInputGradImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
        return output;
    });
    
    mod.add_type<GaussianSamplerDensity<MemorySpace>>("GaussianSamplerDensity", jlcxx::julia_base_type<DensityBase<MemorySpace>>());
    mod.add_type<PullbackDensity<MemorySpace>>("PullbackDensity"      , jlcxx::julia_base_type<DensityBase<MemorySpace>>());
    mod.add_type<PushforwardDensity<MemorySpace>>("PushforwardDensity", jlcxx::julia_base_type<DensityBase<MemorySpace>>());

    mod.add_type<PullbackSampler<MemorySpace>>("PullbackSampler", jlcxx::julia_base_type<SampleGenerator<MemorySpace>>());
    mod.add_type<PushforwardSampler<MemorySpace>>("PushforwardSampler", jlcxx::julia_base_type<SampleGenerator<MemorySpace>>());
    
    // Transport Distribution Factory
    mod.method("CreatePullback", &TransportDistributionFactory::CreatePullback<MemorySpace>);
    mod.method("CreatePushforward", &TransportDistributionFactory::CreatePushforward<MemorySpace>);
    mod.method("CreateGaussianPullback", &TransportDistributionFactory::CreateGaussianPullback<MemorySpace>);
    mod.method("CreateGaussianPushforward", &TransportDistributionFactory::CreateGaussianPushforward<MemorySpace>);
}