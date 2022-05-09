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

        <h3>Usage Examples </h3>

        Conversion from a std::vector of doubles to a a Kokkos array.
        @code{cpp}
        std::vector<double> array;
        // fill in array here....

        Kokkos::View<double*> view = ToKokkos<double>(&array[0], array.size());
        @endcode

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

        <h3>Usage Examples </h3>

        Conversion from a block of memory containing a \f$N\times M\f$ matrix (in a **column**-major layout):
        @code{cpp}
        unsigned int N = 10;
        unsigned int M = 20;
        std::vector<double> array(N*M);
        // fill in matrix here....

        Kokkos::View<double*> view = ToKokkos<double>(&array[0], N, M);
        @endcode

        It is also possible to specify the layout of the data (e.g., row major) by adding an additional template argument.  Here is the conversion from a block of memory containing a \f$N\times M\f$ **row**-major matrix:
        @code{cpp}
        unsigned int N = 10;
        unsigned int M = 20;
        std::vector<double> array(N*M);
        // fill in matrix here....

        Kokkos::View<double*> view = ToKokkos<double,Kokkos::LayoutRight>(&array[0], N, M);
        @endcode

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
    
    /** @brief Converts a column major 2d Eigen::Ref of a matrix to an unmanaged Kokkos view.  
        @ingroup ArrayUtilities
        @details Creates a Kokkos unmanaged view around an existing Eigen object. 
                 Currently only works with objects on the Host.

                Note that this function returns a Kokkos::View with strided layout.  This is different than the typical 
                Kokkos::LayoutLeft and Kokkos::LayoutRight layouts typically used by default.   A list of admissable 
                conversions between layout types can be found in [the Kokkos documentation](https://github.com/kokkos/kokkos/wiki/View#655-conversion-rules-and-function-specialization).
        
        <h4>Usage examples:</h4>
        @code{.cpp}
        Eigen::MatrixXd A;
        // Fill in A... ;

        // Create a 2d view to the matrix
        auto view = MatToKokkos<double>(A);
        @endcode 

        @code{.cpp}
        Eigen::MatrxXd A;
        // Fill in A... ;

        // Create a 2d view to a 10x10 block of the matrix
        auto view = MatToKokkos<double>( A.block(1,2,10,10) );
        @endcode 


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

        <h3>Usage examples:</h3>
        @code{cpp}
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A;
        // Fill in A... ;

        // Create a 2d view to the matrix
        auto view = MatToKokkos<double>(A);
        @endcode 

        @code{cpp}
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A;
        // Fill in A... ;

        // Create a 2d view to a 10x10 block of the matrix
        auto view = MatToKokkos<double>( A.block(1,2,10,10) );
        @endcode 

        @code{cpp}
        Eigen::MatrixXd A;
        // Fill in A... ;

        // Create a 2d view to the transpose of the matrix
        auto view = MatToKokkos<double>( A.transpose() );
        @endcode 

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

        <h4>Usage examples:</h4>
        @code{.cpp}
        Eigen::VectorXd x;
        // Fill in x... ;

        // Create a 1d view to the vector
        auto view = VecToKokkos<double>(x);
        @endcode 

        @code{.cpp}
        Eigen::MatrxXd A;
        // Fill in x... ;

        // Create a 1d view to the first row of the matrix
        auto view = VecToKokkos<double>(A.row(0));
        @endcode 

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


