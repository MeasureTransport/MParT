#include "CommonPybindUtilities.h"

using namespace mpart::python;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);

}