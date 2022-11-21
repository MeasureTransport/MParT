#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/Serialization.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/MapOptions.h"


namespace py = pybind11;
using namespace mpart;
using namespace mpart::binding;

template<>
void mpart::binding::SerializeWrapper<Kokkos::HostSpace>(py::module &m)
{
    m.def("SerializeMapCoeffs",
    [](ParameterizedFunctionBase<Kokkos::HostSpace> const &obj, std::string const &filename) {
        std::ofstream os(filename);
        cereal::BinaryOutputArchive archive(os);
        archive(obj.inputDim, obj.outputDim, obj.numCoeffs);
        save(archive, obj.Coeffs());
    });

    m.def("SerializeFixedMultiIndexSet",
    [](std::shared_ptr<FixedMultiIndexSet<Kokkos::HostSpace>> const &obj, std::string const &filename) {
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

template<>
void mpart::binding::DeserializeWrapper<Kokkos::HostSpace>(py::module &m)
{
    m.def("DeserializeMapCoeffs",
    [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        unsigned int inputDim, outputDim, numCoeffs;
        archive(inputDim, outputDim, numCoeffs);
        Kokkos::View<double*, Kokkos::HostSpace> coeffs ("Map coeffs", numCoeffs);
        load(archive, coeffs);
        return CopyKokkosToVec(coeffs);
    });

    m.def("DeserializeFixedMultiIndexSet",
    [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        std::shared_ptr<FixedMultiIndexSet<Kokkos::HostSpace>> obj {nullptr};
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