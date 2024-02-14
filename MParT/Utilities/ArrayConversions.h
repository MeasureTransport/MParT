#ifndef ARRAYCONVERSIONS_H
#define ARRAYCONVERSIONS_H

#include <Kokkos_Core.hpp>
#include <Eigen/Core>

#include "GPUtils.h"

namespace mpart{

    /** @defgroup ArrayUtilities
        @brief Code for converting between different array types.  Often used in bindings to other languages.
    */

    /** Alias declaration for strided Kokkos matrix type. */
    template<typename ScalarType, typename MemorySpace>
    using StridedMatrix = Kokkos::View<ScalarType**, Kokkos::LayoutStride, MemorySpace>;

    template<typename ScalarType, typename MemorySpace>
    using StridedVector = Kokkos::View<ScalarType*, Kokkos::LayoutStride, MemorySpace>;

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
    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline Kokkos::View<ScalarType*,MemorySpace> ToKokkos(ScalarType* ptr, unsigned int dim)
    {
        return Kokkos::View<ScalarType*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, dim);
    }

    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline Kokkos::View<const ScalarType*,MemorySpace> ToConstKokkos(const ScalarType* const_ptr, unsigned int dim)
    {
        ScalarType* ptr = const_cast<ScalarType*>(const_ptr);
        return Kokkos::View<const ScalarType*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, dim);
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
    template<typename ScalarType, typename LayoutType=Kokkos::LayoutLeft, typename MemorySpace=Kokkos::HostSpace>
    inline Kokkos::View<ScalarType**, LayoutType, MemorySpace> ToKokkos(ScalarType* ptr, unsigned int rows, unsigned int cols)
    {
        return Kokkos::View<ScalarType**, LayoutType, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, rows, cols);
    }

    /** Constructs a Kokkos::View with strided layout from a memory address, shape tuple, and stride tuple. 
        This is primarily used to provide an interface with pytorch in the python bindings.
    */
    template<typename ScalarType, typename MemorySpace=Kokkos::HostSpace>
    StridedMatrix<ScalarType, MemorySpace> ToKokkos(std::tuple<long, std::tuple<int,int>, std::tuple<int,int>> info)    
    {
        ScalarType* ptr = reinterpret_cast<ScalarType*>(std::get<0>(info));

        const int rows = std::get<0>(std::get<1>(info));
        const int cols = std::get<1>(std::get<1>(info));
        const int rowStride = std::get<0>(std::get<2>(info));
        const int colStride = std::get<1>(std::get<2>(info));

        Kokkos::LayoutStride layout(rows, rowStride, cols, colStride);

        Kokkos::View<ScalarType**, Kokkos::LayoutStride, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> output(ptr, layout);
        return output;
    }

    template<typename ScalarType, typename MemorySpace=Kokkos::HostSpace>
    StridedVector<ScalarType, MemorySpace> ToKokkos(std::tuple<long, int, int> info)    
    {
        ScalarType* ptr = reinterpret_cast<ScalarType*>(std::get<0>(info));

        const int length = std::get<1>(info);
        const int stride = std::get<2>(info);

        Kokkos::LayoutStride layout(length, stride);

        Kokkos::View<ScalarType*, Kokkos::LayoutStride, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> output(ptr, layout);
        return output;
    }


    template<typename ScalarType, typename LayoutType=Kokkos::LayoutLeft, typename MemorySpace=Kokkos::HostSpace>
    inline Kokkos::View<const ScalarType**, LayoutType, MemorySpace> ToConstKokkos(const ScalarType* const_ptr, unsigned int rows, unsigned int cols)
    {
        ScalarType* ptr = const_cast<ScalarType*>(const_ptr);
        return Kokkos::View<const ScalarType**, LayoutType, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, rows, cols);
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

    /**
     * @brief Wraps a std::vector in a StridedVector
     *
     * @tparam ScalarType
     * @tparam MemorySpace
     * @param vec
     * @return StridedVector<ScalarType*, MemorySpace>
     */
    template<typename ScalarType, class MemorySpace>
    StridedVector<ScalarType, MemorySpace> VecToKokkos(std::vector<ScalarType> &vec)
    {
        return Kokkos::View<ScalarType*, MemorySpace>(vec.data(), vec.size());
    }

    /**
     * @brief Wraps a const std::vector in a StridedVector
     *
     * @tparam ScalarType
     * @tparam MemorySpace
     * @param vec
     * @return StridedVector<ScalarType*, MemorySpace>
     */
    template<typename ScalarType, class MemorySpace>
    StridedVector<ScalarType, MemorySpace> ConstVecToKokkos(const std::vector<ScalarType> &vec)
    {
        ScalarType* ptr = const_cast<ScalarType*>(vec.data());
        return Kokkos::View<ScalarType*, MemorySpace>(ptr, vec.size());
    }

    /**
     * @brief Wraps a std::vector in a StridedMatrix
     *
     * @tparam ScalarType
     * @tparam MemorySpace
     * @param vec
     * @return Kokkos::View<ScalarType*, MemorySpace>
     */
    template<typename ScalarType, class MemorySpace>
    StridedMatrix<ScalarType, MemorySpace> MatToKokkos(std::vector<ScalarType> &vec, int cols)
    {
        auto rows = vec.size()/cols;
        return Kokkos::View<ScalarType**, MemorySpace>(vec.data(), rows, cols);
    }

    template<typename ScalarType, class MemorySpace>
    StridedVector<const ScalarType, MemorySpace> VecToConstKokkos(std::vector<ScalarType> &vec)
    {
        return Kokkos::View<const ScalarType*, MemorySpace>(vec.data(), vec.size());
    }

    /**
     * @brief Wraps a std::vector in a StridedMatrix
     *
     * @tparam ScalarType
     * @tparam MemorySpace
     * @param vec
     * @return Kokkos::View<ScalarType*, MemorySpace>
     */
    template<typename ScalarType, class MemorySpace>
    StridedMatrix<const ScalarType, MemorySpace> MatToConstKokkos(std::vector<ScalarType> &vec, int cols)
    {
        auto rows = vec.size()/cols;
        return Kokkos::View<const ScalarType**, MemorySpace>(vec.data(), rows, cols);
    }


    /**
    @brief Copies a Kokkos array from device memory to host memory
    @details
    @tparam ViewType A Kokkos View type in device space.
    @param[in] inview A kokkos view ind device space.
    @return A kokkos array in host memory, that is a copy of the inview.
    */
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
    */
    template<class ViewType>
    typename ViewType::HostMirror ToHost(ViewType const& inview){
        typename ViewType::HostMirror outview = Kokkos::create_mirror_view(inview);
        Kokkos::deep_copy (outview, inview);
        return outview;
    }

#if defined(MPART_ENABLE_GPU)

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

    template<typename DeviceMemoryType, typename... OtherTraits>
    StridedMatrix<typename Kokkos::View<OtherTraits...>::non_const_value_type, DeviceMemoryType> ToDevice(Kokkos::View<OtherTraits...> const& inview)
    {
        size_t stride0 = inview.stride_0();
        size_t stride1 = inview.stride_1();

        if(stride0==1){
            Kokkos::View<typename Kokkos::View<OtherTraits...>::non_const_value_type**, Kokkos::LayoutLeft, DeviceMemoryType> outview("Device Copy", inview.extent(0), inview.extent(1));
            Kokkos::deep_copy(outview, inview);
            return outview;
        }else if(stride1==1){
            Kokkos::View<typename Kokkos::View<OtherTraits...>::non_const_value_type**, Kokkos::LayoutRight, DeviceMemoryType> outview("Device Copy", inview.extent(0), inview.extent(1));
            Kokkos::deep_copy(outview, inview);
            return outview;
        }else{
            std::stringstream msg;
            msg << "Cannot copy generally strided matrix to device.  MParT currently only supports copies of view with continguous memory layouts.";
            throw std::runtime_error(msg.str());
        }

    }

    template<typename ScalarType>
    inline Kokkos::View<ScalarType*,DeviceSpace> ToKokkos<ScalarType,DeviceSpace>(ScalarType* ptr, unsigned int dim) {
        Kokkos::View<ScalarType*,DeviceSpace> view = ToKokkos(ptr, dim);
        return ToDevice(view);
    }

    template<typename ScalarType>
    inline Kokkos::View<const ScalarType*,DeviceSpace> ToConstKokkos<ScalarType,DeviceSpace>(ScalarType* ptr, unsigned int dim) {
        Kokkos::View<const ScalarType*,DeviceSpace> view = ToConstKokkos(ptr, dim);
        return ToDevice(view);
    }

    template<typename ScalarType, typename LayoutType = Kokkos::LayoutLeft>
    inline StridedMatrix<ScalarType, DeviceSpace> ToKokkos<ScalarType,LayoutType,DeviceSpace>(ScalarType* ptr, unsigned int rows, unsigned int cols) {
        Kokkos::View<ScalarType*,LayoutType,Kokkos::HostSpace> view = ToKokkos<ScalarType,LayoutType,Kokkos::HostSpace>(ptr, rows, cols);
        return ToDevice(view);
    }

    template<typename ScalarType, typename LayoutType = Kokkos::LayoutLeft>
    inline StridedMatrix<const ScalarType, DeviceSpace> ToConstKokkos<ScalarType,LayoutType,DeviceSpace>(ScalarType* ptr, unsigned int rows, unsigned int cols) {
        Kokkos::View<const ScalarType**,LayoutType,Kokkos::HostSpace> view = ToConstKokkos<ScalarType,LayoutType,Kokkos::HostSpace>(ptr, rows, cols);
        return ToDevice(view);
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
    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline StridedMatrix<ScalarType, MemorySpace> MatToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ref)
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
    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline StridedMatrix<ScalarType, MemorySpace> MatToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ref)
    {
        Kokkos::LayoutStride strides(ref.rows(), ref.outerStride(), ref.cols(), ref.innerStride());

        return Kokkos::View<ScalarType**, Kokkos::LayoutStride, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), strides);
    }

    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline Kokkos::View<const ScalarType**, Kokkos::LayoutRight, MemorySpace> ConstRowMatToKokkos(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
    {
        Kokkos::LayoutStride strides(ref.rows(), ref.innerStride(), ref.cols(), ref.outerStride());
        return Kokkos::View<const ScalarType**, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), ref.rows(), ref.cols());
    }

    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline Kokkos::View<const ScalarType**, Kokkos::LayoutLeft, MemorySpace> ConstColMatToKokkos(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
    {
        return Kokkos::View<const ScalarType**, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), ref.rows(), ref.cols());
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
    template<typename ScalarType, typename MemorySpace = Kokkos::HostSpace>
    inline Kokkos::View<ScalarType*, Kokkos::LayoutStride, MemorySpace> VecToKokkos(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<Eigen::Dynamic>> ref)
    {
        Kokkos::LayoutStride strides(ref.rows(), ref.innerStride());
        return Kokkos::View<ScalarType*, Kokkos::LayoutStride, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), strides);
    }

    #if defined(MPART_ENABLE_GPU)
        template<typename ScalarType>
        StridedMatrix<ScalarType, DeviceSpace> MatToKokkos<ScalarType, DeviceSpace>(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ref)
        {
            auto host = MatToKokkos<ScalarType,Kokkos::HostSpace>(ref);
            return ToDevice(host);
        }

        template<typename ScalarType>
        inline Kokkos::View<const ScalarType**, Kokkos::LayoutRight, DeviceSpace> ConstRowMatToKokkos<ScalarType,DeviceSpace>(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
        {
            auto host = ConstRowMatToKokkos<ScalarType,Kokkos::HostSpace>(ref);
            return ToDevice(host);
        }

        template<typename ScalarType>
        inline Kokkos::View<const ScalarType**, Kokkos::LayoutLeft, DeviceSpace> ConstColMatToKokkos<ScalarType,DeviceSpace>(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
        {
            auto host = ConstColMatToKokkos<ScalarType,Kokkos::HostSpace>(ref);
            return ToDevice(host);
        }

        template<typename ScalarType>
        inline Kokkos::View<ScalarType*, Kokkos::LayoutStride, DeviceSpace> VecToKokkos<ScalarType,DeviceSpace>(Eigen::Ref<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<Eigen::Dynamic>> ref) {
            auto host = VecToKokkos<ScalarType,Kokkos::HostSpace>(ref);
            return ToDevice(host);
        }
    #endif

    /**
       @brief Converts a 1d Kokkos view with **contiguous** memory to an Eigen::Map
       @ingroup ArrayUtilities
       @tparam ScalarType The scalar type stored in the view, typically double, int, or unsigned int.
       @tparam OtherTraits Additional traits, like the memory space, used to define the view.
       @param view The Kokkos::View we wish to wrap.
       @return Eigen::Map<Eigen::VectorXd> A map wrapped around the memory stored by the view.  Note that this map will only be valid as long as the view's memory is being managed.  Segfaults could occur if the map is accessed after the view goes out of scope.  To avoid this, copy the map into a vector or use the CopyKokkosToVec function.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>> KokkosToVec(Kokkos::View<ScalarType*, OtherTraits...> view){
        return Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>>(view.data(), view.extent(0));
    }

    /**
       @brief Converts a 1d Kokkos view with **strided** memory access to an Eigen::Map
       @ingroup ArrayUtilities

       @tparam ScalarType The scalar type stored in the view, typically double, int, or unsigned int.
       @tparam OtherTraits Additional traits, like the memory space, used to define the view.
       @param view The Kokkos::View we wish to wrap.
       @return Eigen::Map<Eigen::VectorXd> A map wrapped around the memory stored by the view.  Note that this map will only be valid as long as the view's memory is being managed.  Segfaults could occur if the map is accessed after the view goes out of scope.  To avoid this, copy the map into a vector or use the CopyKokkosToVec function.
    */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<>> KokkosToVec(Kokkos::View<ScalarType*, Kokkos::LayoutStride, OtherTraits...> view){

        size_t stride;
        view.stride(&stride);
        return Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<>>(view.data(), view.extent(0), Eigen::InnerStride<>(stride));
    }

    /**
       @brief Copies memory in a 1d Kokkos to an Eigen::VectorXd
       @ingroup ArrayUtilities

       @tparam ScalarType The scalar type stored in the view, typically double, int, or unsigned int.
       @tparam OtherTraits Additional traits, like the memory space, used to define the view.
       @param view The Kokkos::View we wish to copy.
       @return Eigen::Matrix<ScalarType,Eigen::Dynamic,1> An Eigen vector with a copy of the contents in the Kokkos view.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Matrix<typename std::remove_const<ScalarType>::type,Eigen::Dynamic,1> CopyKokkosToVec(Kokkos::View<ScalarType*, OtherTraits...> view){
        return KokkosToVec<ScalarType>(view);
    }


    // Template metaprogram for getting the Eigen layout type (e.g., ColMajor or RowMajor) that corresponds to the Kokkos layouttype (e.g., LayoutLeft or LayoutRight)
    template<typename KokkosLayoutType>
    struct LayoutToEigen{
        static constexpr Eigen::StorageOptions Layout = Eigen::RowMajor;
    };
    template<>
    struct LayoutToEigen<Kokkos::LayoutLeft>{
        static constexpr Eigen::StorageOptions Layout =  Eigen::ColMajor;
    };

    // Template Metaprogram for converting a Kokkos view type to an Eigen Matrix type
    template<typename ViewType>
    struct ViewToEigen{
    };
    template<typename ViewType>
    struct ConstViewToEigen{
    };
    template<typename ScalarType, typename... OtherTraits>
    struct ViewToEigen<Kokkos::View<ScalarType*,OtherTraits...>>{
        using Type = typename Eigen::Matrix<ScalarType,Eigen::Dynamic,1>;
    };
    template<typename ScalarType, typename... OtherTraits>
    struct ConstViewToEigen<Kokkos::View<ScalarType*,OtherTraits...>>{
        using Type = const typename Eigen::Matrix<typename std::remove_const<ScalarType>::type,Eigen::Dynamic,1>;
    };
    template<typename ScalarType, typename... OtherTraits>
    struct ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>{
        using Type = typename Eigen::Matrix<typename std::remove_const<ScalarType>::type, Eigen::Dynamic, Eigen::Dynamic, LayoutToEigen<typename Kokkos::View<ScalarType*,OtherTraits...>::array_layout>::Layout>;
    };
    template<typename ScalarType, typename... OtherTraits>
    struct ConstViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>{
        using Type = const typename Eigen::Matrix<typename std::remove_const<ScalarType>::type, Eigen::Dynamic, Eigen::Dynamic, LayoutToEigen<typename Kokkos::View<ScalarType*,OtherTraits...>::array_layout>::Layout>;
    };


    /**
     @brief Creates an Eigen::Map around existing memory in a 2D Kokkos::View with either LayoutLeft or LayoutRight layouts.

     @tparam ScalarType The scalar type stored in the Kokkos::View (e.g., double, unsigned int)
     @tparam OtherTraits Other properties of the Kokkos::View
     @param view The Kokkos matrix we wish wrap with an Eigen view.
     @return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::OuterStride<>>  And Eigen::Map with dynamic outer stride and contiguous inner stride.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::OuterStride<>> KokkosToMat(Kokkos::View<ScalarType**,OtherTraits...> view)
    {
        size_t strides[2];
        view.stride(strides);
        assert((strides[0]==1)||(strides[1]==1));
        return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::OuterStride<>>(view.data(), view.extent(0), view.extent(1), Eigen::OuterStride<>(std::max(strides[0],strides[1])));
    }
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<typename ConstViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::OuterStride<>> ConstKokkosToMat(Kokkos::View<const ScalarType**,OtherTraits...> view)
    {
        size_t strides[2];
        view.stride(strides);
        assert((strides[0]==1)||(strides[1]==1));
        return Eigen::Map<typename ConstViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::OuterStride<>>(view.data(), view.extent(0), view.extent(1), Eigen::OuterStride<>(std::max(strides[0],strides[1])));
    }
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<typename ConstViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>> ConstKokkosToMat(Kokkos::View<const ScalarType**, Kokkos::LayoutStride, OtherTraits...> view){

        size_t strides[2];
        view.stride(strides);
        return Eigen::Map<typename ConstViewToEigen<Kokkos::View<ScalarType**, Kokkos::LayoutStride, OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>(view.data(), view.extent(0), view.extent(1), Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(strides[0],strides[1]));
    }


    /**
     @brief Creates an Eigen::Map around existing memory in a 2D Kokkos::View with the general LayoutStride layouts.

     @details
      Note that no memory is copied here, the Eigen::Map simply wraps around the memory used by the Kokkos::View.  If you
      change a value of the map, it will therefore change a value in the view.  If this isn't what you want, you can
      perform a deep copy of the view using the mpart::CopyKokkosToMat function, which has the same arguments.

     <h3>Typical usage</h3>
     @code{cpp}
     Kokkos::View<double**, Kokkos::HostSpace> view("View", N, M);
     auto map = KokkosToMat(view);

     map(1,1) = 1;
     std::cout << "The value of the view and map should be the same:  " << view(1,1) << " vs " << map(1,1) << std::endl;
     @endcode

     @tparam ScalarType The scalar type stored in the Kokkos::View (e.g., double, unsigned int)
     @tparam OtherTraits Other properties of the Kokkos::View
     @param view The Kokkos matrix we wish wrap with an Eigen view.
     @return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>  And Eigen::Map with dynamic outer stride and contiguous inner stride.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>> KokkosToMat(Kokkos::View<ScalarType**, Kokkos::LayoutStride, OtherTraits...> view){

        size_t strides[2];
        view.stride(strides);
        return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**, Kokkos::LayoutStride, OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>(view.data(), view.extent(0), view.extent(1), Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(strides[0],strides[1]));
    }

    /**
     @brief Copies the contents of a Kokkos view into an Eigen matrix with the same memory layout (row major or column major).

     @details
      If you do not need to copy the memory, the mpart::KokkosToMat function, which returns an Eigen::Map, will likely be more
      efficient because it does not copy anything.

     <h3>Typical usage</h3>
     For a column-major view:
     @code{cpp}
     Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> view("View", N, M);
     Eigen::MatrixXd mat = CopyKokkosToMat(view);
     @endcode

     For a row-major view:
     @code{cpp}
     Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> view("View", N, M);
     Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> map = CopyKokkosToMat(view);
     @endcode

     @tparam ScalarType The scalar type stored in the Kokkos::View (e.g., double, unsigned int)
     @tparam OtherTraits Other properties of the Kokkos::View
     @param view The Kokkos matrix we wish wrap with an Eigen view.
     @return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>  And Eigen::Map with dynamic outer stride and contiguous inner stride.
     */
    template<typename ViewType>
    inline typename ViewToEigen<ViewType>::Type CopyKokkosToMat(ViewType view){
        return KokkosToMat(view);
    }

}

#endif  // #ifndef ARRAYCONVERSIONS_H


