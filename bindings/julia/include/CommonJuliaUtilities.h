#ifndef MPART_COMMONJULIAUTILITIES_H
#define MPART_COMMONJULIAUTILITIES_H

#include "jlcxx/jlcxx.hpp"

#include <string>
#include <vector>
#include <chrono>

namespace mpart{
namespace julia{
    /*
    The KokkosGuard class, GetKokkosGuard function, and KokkosCustomPointer class are used to ensure that
    Kokkos::finalize is called when all wrapped Kokkos variables go out of scope.  In the Julia bindings,
    all classes using Kokkos should be stored using the KokkosCustomPointer class.
    */
    struct KokkosGuard
    {
    KokkosGuard();
    ~KokkosGuard();
    };

    std::shared_ptr<KokkosGuard> GetKokkosGuard();

    template <typename T>
    class KokkosCustomPointer {
        std::shared_ptr<KokkosGuard> guard;
        std::shared_ptr<T> impl;
    public:
        KokkosCustomPointer( ) : guard(GetKokkosGuard()) {};
        explicit KokkosCustomPointer(T *p) : guard(GetKokkosGuard()), impl(p) {}
        T* get() const { return impl.get(); }
    };
} // namespace mpart
} // namespace julia 


// Note that this macro must be called in the jlcxx namespace, which is why there are two separate namespace blocks in this file
namespace jlcxx{
  template<typename T> struct IsSmartPointerType<mpart::julia::KokkosCustomPointer<T>> : std::true_type { };
  template<typename T> struct ConstructorPointerType<mpart::julia::KokkosCustomPointer<T>> { typedef std::shared_ptr<T> type; };
}


namespace mpart{
namespace julia{

/** KokkosRuntime is used to control when Kokkos::finalize is called.  It also provides a mechanism
    for measuring the time that has elapsed since KokkosInit was called.
*/
class KokkosRuntime
{
public:
    KokkosRuntime();

    double ElapsedTime() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};


/** Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv. */
KokkosRuntime KokkosInit(std::vector<std::string> args);

/** Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv. */
KokkosRuntime KokkosInit(pybind11::dict opts);

/**
   @brief Adds the pybind11 bindings to the existing module pybind11 module m. 
   @param m pybind11 module
 */
void CommonUtilitiesWrapper(pybind11::module &m);

} // namespace python
} // namespace mpart



#endif // MPART_COMMONJULIAUTILITIES_H