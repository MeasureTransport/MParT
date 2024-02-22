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

template<typename MemorySpace>
void mpart::binding::DeserializeWrapper<MemorySpace>(py::module &m)
{
    m.def("DeserializeMap",
    [](std::string const &filename) {
        std::ifstream is(filename);
        cereal::BinaryInputArchive archive(is);
        unsigned int inputDim, outputDim, numCoeffs;
        archive(inputDim, outputDim, numCoeffs);
        Kokkos::View<double*, MemorySpace> coeffs ("Map coeffs", numCoeffs);
        load(archive, coeffs);
        return std::make_tuple(inputDim, outputDim, CopyKokkosToVec(coeffs));
    });
}

template void mpart::binding::DeserializeWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::DeserializeWrapper<mpart::DeviceSpace>(py::module&);
#endif