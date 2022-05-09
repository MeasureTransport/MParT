#include "CommonUtilities.h"

#include <Kokkos_Core.hpp>

using namespace mpart::binding;


KokkosGuard::KokkosGuard()  
{ 

}

KokkosGuard::~KokkosGuard() 
{ 
    Kokkos::finalize(); 
}

std::shared_ptr<KokkosGuard> mpart::binding::GetKokkosGuard()
{
  static std::weak_ptr<KokkosGuard> kokkos_guard_;
  auto shared = kokkos_guard_.lock();
  if (!shared)
  {
    shared = std::make_shared<KokkosGuard>();
    kokkos_guard_ = shared;
  }
  return shared;
}


KokkosRuntime::KokkosRuntime() : start(std::chrono::high_resolution_clock::now()) {};

double KokkosRuntime::ElapsedTime() const{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count() / 1e6;
};

// Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv.
KokkosRuntime mpart::binding::KokkosInit(std::vector<std::string> args) {

    std::vector<char *> cstrs;
    cstrs.reserve(args.size());
    for (auto &s : args) cstrs.push_back(const_cast<char *>(s.c_str()));

    int size = cstrs.size();
    Kokkos::initialize(size, cstrs.data());

    return KokkosRuntime();
};