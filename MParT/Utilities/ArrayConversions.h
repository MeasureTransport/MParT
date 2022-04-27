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
    Kokkos::View<ScalarType*> ToKokkos(ScalarType* ptr, unsigned int dim)
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
    Kokkos::View<ScalarType**, LayoutType> ToKokkos(ScalarType* ptr, unsigned int rows, unsigned int cols)
    {   
        return Kokkos::View<ScalarType**, LayoutType, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, rows, cols);
    }


    /**
     * @brief Returns a copy of a one dimensional Kokkos::View in the form of a std::vector.
     * 
     * @tparam ScalarType 
     * @param view 
     * @return std::vector<std::remove_const<ScalarType>::type> 
     */
    template<typename ScalarType, class... ViewTraits>
    std::vector<typename std::remove_const<ScalarType>::type> KokkosToStd(Kokkos::View<ScalarType*,ViewTraits...> const& view)
    {
        std::vector<typename std::remove_const<ScalarType>::type> output(view.extent(0));
        for(unsigned int i=0; i<view.extent(0); ++i)
            output[i] = view(i);
        return output;
    }
}

#endif  // #ifndef ARRAYCONVERSIONS_H


