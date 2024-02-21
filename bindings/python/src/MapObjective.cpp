#include "CommonPybindUtilities.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;
using namespace mpart;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::MapObjectiveWrapper(py::module &m) {
    std::string t1Name= "MapObjective";
    std::string t2Name = "KLObjective";
    std::string mName = "CreateGaussianKLObjective";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) {
        t1Name = "d" + t1Name;
        t2Name = "d" + t2Name;
        mName = "d" + mName;
    }

    py::class_<MapObjective<MemorySpace>, std::shared_ptr<MapObjective<MemorySpace>>>(m, t1Name.c_str())
        .def("TestError", &KLObjective<MemorySpace>::TestError)
        .def("TrainError", &KLObjective<MemorySpace>::TrainError)
    ;

    py::class_<KLObjective<MemorySpace>, MapObjective<MemorySpace>, std::shared_ptr<KLObjective<MemorySpace>>>(m, t2Name.c_str());
    m.def(mName.c_str(), [](Eigen::Ref<Eigen::MatrixXd> &train, unsigned int dim){
            StridedMatrix<const double, MemorySpace> trainView = RowMatToKokkos<double, MemorySpace>(train);
            Kokkos::View<double**,MemorySpace> storeTrain ("Training data store", trainView.extent(0), trainView.extent(1));
            Kokkos::deep_copy(storeTrain, trainView);
            trainView = storeTrain;
            return ObjectiveFactory::CreateGaussianKLObjective(trainView, dim);
        }, "train"_a, "dim"_a = 0)
        .def(mName.c_str(), [](Eigen::Ref<Eigen::MatrixXd> &trainEig, Eigen::Ref<Eigen::MatrixXd> &testEig, unsigned int dim){
            StridedMatrix<const double, MemorySpace> trainView = MatToKokkos<double, MemorySpace>(trainEig);
            StridedMatrix<const double, MemorySpace> testView = MatToKokkos<double, MemorySpace>(testEig);
            Kokkos::View<double**,MemorySpace> storeTrain ("Training data store", trainView.extent(0), trainView.extent(1));
            Kokkos::View<double**,MemorySpace> storeTest ("Testing data store", testView.extent(0), testView.extent(1));
            Kokkos::deep_copy(storeTrain, trainView);
            Kokkos::deep_copy(storeTest, testView);
            trainView = storeTrain;
            testView = storeTest;
            return ObjectiveFactory::CreateGaussianKLObjective(trainView, testView, dim);
        }, "train"_a, "test"_a, "dim"_a = 0)
    ;
}

template void mpart::binding::MapObjectiveWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::MapObjectiveWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU
