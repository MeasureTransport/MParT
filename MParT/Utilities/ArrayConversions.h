#ifndef ARRAYCONVERSIONS_H
#define ARRAYCONVERSIONS_H

#include <Kokkos_Core.hpp>

namespace mpart{

    /** @defgroup ArrayUtilities
        @brief Code for converting between different array types.  Often used in bindings to other languages.
    */
    
    /** @brief Converts a pointer to a 1d unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around a preallocated block of memory.  
                 The unmanaged view will not free the memory so all allocations and deallocations 
                 need to be handled manually (or via another object like an Eigen::Matrix).  Currently
                 only works for memory on the Host device.  
        @todo Extend to work with device memory.

        @param[in] ptr A pointer to a block of memory defining the array.  Note 
        @param[in] dim The length of the array.
        @return A Kokkos view wrapping around the memory pointed to by ptr.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType>
    Kokkos::View<ScalarType*, Kokkos::HostSpace> ToKokkos(ScalarType* ptr, unsigned int dim)
    {
        return Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, dim);
    }

    /** @brief Converts a pointer to a 2d unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around a preallocated block of memory. 
                 The unmanaged view will not free the memory so all allocations and deallocations 
                 need to be handled manually (or via another object like an Eigen::Matrix).  Currently
                 only works for memory on the Host device.  
        @todo Extend to work with device memory.

        @param[in] ptr A pointer to a block of memory defining the array.  Note that this array must have at least rows*cols allocated after this pointer or a segfault is likely.
        @param[in] rows The number of rows in the matrix.
        @param[in] cols The number of columns in the matrix.
        @return A 2D Kokkos view wrapping around the memory pointed to by ptr.
        @tparam LayoutType A kokkos layout type dictating whether the memory in ptr is organized in column major format or row major format.   If LayoutType is Kokkos::LayoutRight, the data is treated in row major form.  If LayoutType is Kokkos::LayoutLeft, the data is treated in column major form.  Defaults to Kokkos::LayoutLeft.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType, typename LayoutType=Kokkos::LayoutLeft>
    Kokkos::View<ScalarType**, LayoutType,Kokkos::HostSpace> ToKokkos(ScalarType* ptr, unsigned int rows, unsigned int cols)
    {   
        return Kokkos::View<ScalarType**, LayoutType, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, rows, cols);
    }

    /**
    @brief Copies a Kokkos array from device memory to host memory
    @details
    @param[in] inview A kokkos array in device memory.
    @return A kokkos array in host memory.  Note that the layout (row-major or col-major) might be different than the default on the Host.  The layout will match the device's default layout.
    */
    template<typename DeviceMemoryType, typename ScalarType>
    Kokkos::View<ScalarType, Kokkos::HostSpace> ToHost(Kokkos::View<ScalarType,DeviceMemoryType> const& inview){
        typename Kokkos::View<ScalarType>::HostMirror outview = Kokkos::create_mirror_view(inview);
        Kokkos::deep_copy (outview, inview);
        return outview;
    }

    /**
    @brief Copies a range of elements from a Kokkos array in device to host memory
    @details
    Typical usage for a 1d array is something like:
    @code{cpp}
    const unsigned int N = 10;
    Kokkos::View<double*,Kokkos::CudaSpace> deviceView("Some stuff on the device", N);
    Kokkos::View<double*,Kokkos::HostSpace> hostView = ToHost(deviceView, std::make_pair(2, 3)); // Similar to python notation: deviceView[2:3]
    @endcode
    Similarly, usage for a 2d array might be something like
    @code{cpp}
    const unsigned int N1 = 10;
    const unsigned int N2 = 100;
    Kokkos::View<double**,Kokkos::CudaSpace> deviceView("Some stuff on the device", N1, N2);
    Kokkos::View<double*,Kokkos::HostSpace> hostView = ToHost(deviceView, std::make_pair(2,4), std::make_pair(3,50) ); // Similar to python notation: deviceView[2:4,3:50]
    @endcode
    Extracting an entire row or column of a Kokkos::View can be accomplished with the Kokkos::All() function
    @code{cpp}
    const unsigned int N1 = 10;
    const unsigned int N2 = 100;
    Kokkos::View<double**,Kokkos::CudaSpace> deviceView("Some stuff on the device", N1, N2);
    Kokkos::View<double*,Kokkos::HostSpace> hostView = ToHost(deviceView, 2, Kokkos::All() ); // Similar to python notation: deviceView[2,:]
    @endcode
    
    @param[in] inview A kokkos array in device memory.
    @param[in] sliceParams One or more parameters defining a Kokkos::subview.  See the [Kokkos Subview documentation](https://github.com/kokkos/kokkos/wiki/Subviews#112-how-to-take-a-subview) for more details.
    @return A kokkos array in host memory.  Note that the layout (row-major or col-major) might be different than the default on the Host.  The layout will match the device's default layout.

    @tparam DeviceMemoryType The memory space (e.g., Kokkos::CudaSpace) or the device
    @tparam ScalarType The type and dimension of the Kokkos::View (e.g., double*, double**, or int*)
    @tparam SliceTypes A variadic parameter pack containing options for constructing a Kokkos::subview of the device view.
    */
    template<typename DeviceMemoryType, typename ScalarType, class... SliceTypes>
    Kokkos::View<ScalarType, Kokkos::HostSpace> ToHost(Kokkos::View<ScalarType,DeviceMemoryType> const& inview, SliceTypes... sliceParams){
        auto subview = Kokkos::subview(inview, sliceParams...); // Construct the subview
        typename Kokkos::View<ScalarType>::HostMirror outview = Kokkos::create_mirror_view(subview);
        Kokkos::deep_copy (outview, subview);
        return outview;
    }


#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

    /**
    @brief Copies a 1d Kokkos array from host memory to device memory.
    @details
    @param[in] inview A kokkos array in host memory. 
    @return A kokkos array in device memory. 
    */
    template<typename DeviceMemoryType,typename ScalarType>
    Kokkos::View<ScalarType*, DeviceMemoryType> ToDevice(Kokkos::View<ScalarType*, Kokkos::HostSpace> const& inview){

        Kokkos::View<ScalarType*, DeviceMemoryType> outview("Device Copy", inview.extent(0));
        Kokkos::deep_copy(outview, inview);
        return outview;

    }

    template<typename DeviceMemoryType, typename ScalarType, class... OtherTraits>
    Kokkos::View<ScalarType**, DeviceMemoryType> ToDevice(Kokkos::View<ScalarType**, OtherTraits...>const& inview){

        Kokkos::View<ScalarType**, DeviceMemoryType> outview("Device Copy", inview.extent(0), inview.extent(1));
        Kokkos::deep_copy(outview, inview);
        return outview;
    }

    template<typename DeviceMemoryType,typename ScalarType>
    Kokkos::View<ScalarType*, DeviceMemoryType> ToDevice(Kokkos::View<ScalarType*, DeviceMemoryType> const& inview){
        return inview;
    }

    template<typename DeviceMemoryType,typename ScalarType>
    Kokkos::View<ScalarType**, DeviceMemoryType> ToDevice(Kokkos::View<ScalarType**, DeviceMemoryType> const& inview){
        return inview;
    }


#endif

}

#endif  // #ifndef ARRAYCONVERSIONS_H


