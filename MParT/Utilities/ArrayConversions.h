#ifndef ARRAYCONVERSIONS_H
#define ARRAYCONVERSIONS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Layout.hpp>
#include <Eigen/Core>

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

        @param[in] ptr A pointer to a block of memory defining the array.  Note 
        @param[in] dim The length of the array.
        @return A Kokkos view wrapping around the memory pointed to by ptr.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType>
    inline Kokkos::View<ScalarType*,Kokkos::HostSpace> ToKokkos(ScalarType* ptr, unsigned int dim)
    {
        return Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, dim);
    }

    /** @brief Converts a pointer to a 2d unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around a preallocated block of memory. 
                 The unmanaged view will not free the memory so all allocations and deallocations 
                 need to be handled manually (or via another object like an Eigen::Matrix).  Currently
                 only works for memory on the Host device.  

        @param[in] ptr A pointer to a block of memory defining the array.  Note that this array must have at least rows*cols allocated after this pointer or a segfault is likely.
        @param[in] rows The number of rows in the matrix.
        @param[in] cols The number of columns in the matrix.
        @return A 2D Kokkos view wrapping around the memory pointed to by ptr.
        @tparam LayoutType A kokkos layout type dictating whether the memory in ptr is organized in column major format or row major format.   If LayoutType is Kokkos::LayoutRight, the data is treated in row major form.  If LayoutType is Kokkos::LayoutLeft, the data is treated in column major form.  Defaults to Kokkos::LayoutLeft.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType, typename LayoutType=Kokkos::LayoutLeft>
    inline Kokkos::View<ScalarType**, LayoutType, Kokkos::HostSpace> ToKokkos(ScalarType* ptr, unsigned int rows, unsigned int cols)
    {   
        return Kokkos::View<ScalarType**, LayoutType, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, rows, cols);
    }

    /** @brief Converts a column major 2d Eigen::Ref of a matrix to an unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around an existing Eigen object. 
                 Currently only works with objects on the Host.

        @param[in] ref The reference to the eigen reference.
        @return A 2D Kokkos unmanaged view wrapping the same memory as the eigen ref and using the same strides as the eigen object.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType>
    inline Kokkos::View<ScalarType**, Kokkos::LayoutStride, Kokkos::HostSpace> MatToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ref)
    {   
        Kokkos::LayoutStride strides(ref.rows(), ref.innerStride(), ref.cols(), ref.outerStride());

        return Kokkos::View<ScalarType**, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), strides);
    }

    /** @brief Converts a row major 2d Eigen::Ref of a matrix to an unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around an existing Eigen object. 
                 Currently only works with objects on the Host.

        @param[in] ref The reference to the eigen reference.
        @return A 2D Kokkos unmanaged view wrapping the same memory as the eigen ref and using the same strides as the eigen object.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType>
    inline Kokkos::View<ScalarType**, Kokkos::LayoutStride, Kokkos::HostSpace> MatToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ref)
    {   
        Kokkos::LayoutStride strides(ref.rows(), ref.outerStride(), ref.cols(), ref.innerStride());

        return Kokkos::View<ScalarType**, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), strides);
    }

    /** @brief Converts a 1d Eigen::Ref of a vector to an unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around an existing Eigen object. 
                 Currently only works with objects on the Host.

        @param[in] ref The reference to the eigen reference.
        @return A 1d Kokkos unmanaged view wrapping the same memory as the eigen ref and using the same stride as the eigen object.
        @tparam ScalarType The scalar type, typically double, int, or unsigned int.
    */
    template<typename ScalarType>
    inline Kokkos::View<ScalarType*, Kokkos::LayoutStride, Kokkos::HostSpace> VecToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<Eigen::Dynamic>> ref)
    {   
        Kokkos::LayoutStride strides(ref.rows(), ref.innerStride());
        return Kokkos::View<ScalarType*, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), strides);
    }

}

#endif  // #ifndef ARRAYCONVERSIONS_H


