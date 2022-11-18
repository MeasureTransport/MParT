#include <cereal/archives/binary.hpp>
#include <fstream>
#include <pybind11/pybind11.h>

#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/Serialization.h"


namespace py = pybind11;
using namespace mpart;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::SerializeWrapper(py::module &m)
{
    m.def(std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "SerializeParameterFunctionBase" : "dSerializeParameterFunctionBase",
    [](ParameterizedFunctionBase<MemorySpace> const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj.inputDim, obj.outputDim, obj.numCoeffs);
        StridedVector<const double,MemorySpace> coeffs = obj.Coeffs();
        save(archive, coeffs);
    });
}

template<typename MemorySpace>
void mpart::binding::DeserializeWrapper(py::module &m)
{
    m.def(std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "DeserializeMapCoeffs" : "dDeserializeMapCoeffs",
    [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        unsigned int inputDim, outputDim, numCoeffs;
        archive(inputDim, outputDim, numCoeffs);
        StridedVector<double, MemorySpace> coeffs = Kokkos::View<double*, MemorySpace>("Map coeffs", numCoeffs);
        load(archive, coeffs);
        return coeffs;
    });
}


template void mpart::binding::SerializeWrapper<Kokkos::HostSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::SerializeWrapper<mpart::DeviceSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU