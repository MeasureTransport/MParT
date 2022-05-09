#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace mpart{
namespace binding{
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
        T& operator*() const{return *impl;};

        T* get() const { return impl.get(); }
    };

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

} // namespace mpart
} // namespace binding