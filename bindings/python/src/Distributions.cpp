#include "CommonPybindUtilities.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/Distribution.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/Distributions/PullbackDensity.h"
#include "MParT/Distributions/PullbackSampler.h"
#include "MParT/Distributions/PushforwardDensity.h"
#include "MParT/Distributions/PushforwardSampler.h"
#include "MParT/Distributions/TransportDistributionFactory.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;
using namespace mpart;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::DistributionsWrapper(py::module &m) {
    py::class_<DensityBase<MemorySpace>,std::shared_ptr<DensityBase<MemorySpace>>>(m, "DensityBase")
        .def("LogDensity", static_cast<Eigen::VectorXd (DensityBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&DensityBase<MemorySpace>::LogDensity))
        .def("LogDensityInputGrad", static_cast<Eigen::RowMatrixXd (DensityBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&DensityBase<MemorySpace>::LogDensityInputGrad))
    ;
    py::class_<SampleGenerator<MemorySpace>,std::shared_ptr<SampleGenerator<MemorySpace>>>(m, "SampleGenerator");
    py::class_<Distribution<MemorySpace>,std::shared_ptr<Distribution<MemorySpace>>>(m, "Distribution")
        .def("LogDensity", static_cast<Eigen::VectorXd (Distribution<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&Distribution<MemorySpace>::LogDensity))
        .def("LogDensityInputGrad", static_cast<Eigen::RowMatrixXd (Distribution<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&Distribution<MemorySpace>::LogDensityInputGrad))
    ;
    py::class_<GaussianSamplerDensity<MemorySpace>, DensityBase<MemorySpace>, SampleGenerator<MemorySpace>, std::shared_ptr<GaussianSamplerDensity<MemorySpace>>>(m, "GaussianSamplerDensity");

    py::class_<PullbackDensity<MemorySpace>,DensityBase<MemorySpace>,std::shared_ptr<PullbackDensity<MemorySpace>>>(m, "PullbackDensity"      );
    py::class_<PushforwardDensity<MemorySpace>,DensityBase<MemorySpace>,std::shared_ptr<PushforwardDensity<MemorySpace>>>(m, "PushforwardDensity");

    py::class_<PullbackSampler<MemorySpace>,SampleGenerator<MemorySpace>,std::shared_ptr<PullbackSampler<MemorySpace>>>(m, "PullbackSampler"      );
    py::class_<PushforwardSampler<MemorySpace>,SampleGenerator<MemorySpace>,std::shared_ptr<PushforwardSampler<MemorySpace>>>(m, "PushforwardSampler");
    
    // Transport Distribution Factory
    m.def("CreatePullback", &TransportDistributionFactory::CreatePullback<MemorySpace>);
    m.def("CreatePushforward", &TransportDistributionFactory::CreatePushforward<MemorySpace>);
    m.def("CreateGaussianPullback", &TransportDistributionFactory::CreateGaussianPullback<MemorySpace>);
    m.def("CreateGaussianPushforward", &TransportDistributionFactory::CreateGaussianPushforward<MemorySpace>);
}

template void mpart::binding::DistributionsWrapper<Kokkos::HostSpace>(py::module&);