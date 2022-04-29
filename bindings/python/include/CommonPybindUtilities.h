#ifndef MPART_COMMONPYBINDUTILITIES_H
#define MPART_COMMONPYBINDUTILITIES_H

#include <pybind11/pybind11.h>

#include <string>
#include <vector>
#include <chrono>

#include "../../utils/include/CommonUtilities.h"

namespace mpart{
namespace binding{
/** Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv. */
KokkosRuntime KokkosInit(pybind11::dict opts);

/**
   @brief Adds the pybind11 bindings to the existing module pybind11 module m. 
   @param m pybind11 module
 */
void CommonUtilitiesWrapper(pybind11::module &m);

void MultiIndexWrapper(pybind11::module &m);

} // namespace binding
} // namespace mpart




#endif 