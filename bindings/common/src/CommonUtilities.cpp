#include "CommonUtilities.h"

#include <Kokkos_Core.hpp>
#include "MParT/Initialization.h"

// Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv.
void mpart::binding::Initialize(std::vector<std::string> args) {

    std::vector<char *> cstrs;
    cstrs.reserve(args.size());
    for (auto &s : args) cstrs.push_back(const_cast<char *>(s.c_str()));

    int size = cstrs.size();
    mpart::Initialize(size, cstrs.data());
};