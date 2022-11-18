#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include <pybind11/pybind11.h>

#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/Serialization.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/MapOptions.h"


namespace py = pybind11;
using namespace mpart;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::SerializeWrapper(py::module &m)
{
    m.def(std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "SerializeMapCoeffs" : "dSerializeMapCoeffs",
    [](ParameterizedFunctionBase<MemorySpace> const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj.inputDim, obj.outputDim, obj.numCoeffs);
        save(archive, obj.Coeffs());
    });

    m.def(std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "SerializeFixedMultiIndexSet" : "dSerializeFixedMultiIndexSet",
    [](std::shared_ptr<FixedMultiIndexSet<MemorySpace>> const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj);
    });

    m.def("SerializeMapOptions", [](MapOptions const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj);
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
        Kokkos::View<double*, MemorySpace> coeffs ("Map coeffs", numCoeffs);
        load(archive, coeffs);
        return KokkosToVec(coeffs);
    });

    m.def(std::is_same_v<MemorySpace, Kokkos::HostSpace> ? "DeserializeFixedMultiIndexSet" : "dDeserializeFixedMultiIndexSet",
    [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        std::shared_ptr<FixedMultiIndexSet<MemorySpace>> obj {nullptr};
        archive(obj);
        return obj;
    });

    m.def("DeserializeMapOptions", [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        MapOptions obj;
        archive(obj);
        return obj;
    });
}


template void mpart::binding::SerializeWrapper<Kokkos::HostSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::SerializeWrapper<mpart::DeviceSpace>(py::module&);
template void mpart::binding::DeserializeWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU