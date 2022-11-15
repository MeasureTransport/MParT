#include <cereal/archives/binary.hpp>
#include <fstream>
#include <pybind11/pybind11.h>

#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"


namespace py = pybind11;
using namespace mpart::binding;

template<typename T>
void SerializeWrapperType(py::module &m) {
    m.def("Serialize", [](T const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj);
    });
}

template<typename MemorySpace>
void mpart::binding::SerializeWrapper(py::module &m)
{
    SerializeWrapperType<ParameterizedFunctionBase<MemorySpace>>(m);
}

template<typename MemorySpace>
void mpart::binding::DeserializeWrapper(py::module &m)
{
    std::string suffix = std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "" : "Device";
    m.def("DeserializeParameterizedFunctionBase" + suffix, [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        mpart::ParameterizedFunctionBase<Kokkos::HostSpace> map;
        archive(map);
        return map;
    });
}


template void mpart::binding::SerializeWrapper<Kokkos::HostSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::SerializeWrapper<mpart::DeviceSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU