#ifndef MPART_COMMONPYBINDUTILITIES_H
#define MPART_COMMONPYBINDUTILITIES_H


#include <pybind11/pybind11.h>

#include <string>
#include <vector>
#include <chrono>

namespace mpart{
namespace python{
    /*
    The KokkosGuard class, GetKokkosGuard function, and KokkosCustomPointer class are used to ensure that
    Kokkos::finalize is called when all wrapped Kokkos variables go out of scope.  In the python bindings,
    all classes using Kokkos should be stored using the KokkosCustomPointer class.  This can be passed as a 
    second template argument to py::class_
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
        T** operator&() { throw std::logic_error("Call of overloaded operator& is not expected"); }
    };
} // namespace mpart
} // namespace python 


// Note that this macro must be called in the top level namespace, which is why there are two separate namespace blocks in this file
PYBIND11_DECLARE_HOLDER_TYPE(T, mpart::python::KokkosCustomPointer<T>);


namespace mpart{
namespace python{

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




#endif 