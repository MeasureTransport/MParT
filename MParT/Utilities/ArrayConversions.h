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

    template<typename ScalarType>
    inline Kokkos::View<const ScalarType**, Kokkos::LayoutRight, Kokkos::HostSpace> ConstRowMatToKokkos(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
    {   
        Kokkos::LayoutStride strides(ref.rows(), ref.innerStride(), ref.cols(), ref.outerStride());
        return Kokkos::View<const ScalarType**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), ref.rows(), ref.cols());
    }

    template<typename ScalarType>
    inline Kokkos::View<const ScalarType**, Kokkos::LayoutLeft, Kokkos::HostSpace> ConstColMatToKokkos(Eigen::Ref<const Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>> const& ref)
    {   
        return Kokkos::View<const ScalarType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> (ref.data(), ref.rows(), ref.cols());
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

    /**
       @brief Converts a 1d Kokkos view with **contiguous** memory to an Eigen::Map
       @ingroup ArrayUtilities
       @tparam ScalarType The scalar type stored in the view, typically double, int, or unsigned int.
       @tparam OtherTraits Additional traits, like the memory space, used to define the view.
       @param view The Kokkos::View we wish to wrap.
       @return Eigen::Map<Eigen::VectorXd> A map wrapped around the memory stored by the view.  Note that this map will only be valid as long as the view's memory is being managed.  Segfaults could occur if the map is accessed after the view goes out of scope.  To avoid this, copy the map into a vector or use the CopyKokkosToVec function.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>> KokkosToVec(Kokkos::View<ScalarType*, OtherTraits...> const& view){
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
    inline Eigen::Map<Eigen::Matrix<ScalarType,Eigen::Dynamic,1>, 0, Eigen::InnerStride<>> KokkosToVec(Kokkos::View<ScalarType*, Kokkos::LayoutStride, OtherTraits...> const& view){
        
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
    inline Eigen::Matrix<ScalarType,Eigen::Dynamic,1> CopyKokkosToVec(Kokkos::View<ScalarType*, OtherTraits...> const& view){
        return KokkosToVec(view);
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
    template<typename ScalarType, typename... OtherTraits>
    struct ViewToEigen<Kokkos::View<ScalarType*,OtherTraits...>>{
        using Type = typename Eigen::Matrix<ScalarType,Eigen::Dynamic,1>;
    };
    template<typename ScalarType, typename... OtherTraits>
    struct ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>{
        using Type = typename Eigen::Matrix<ScalarType,Eigen::Dynamic, Eigen::Dynamic, LayoutToEigen<typename Kokkos::View<ScalarType*,OtherTraits...>::array_layout>::Layout>;
    };


    /**
     @brief Creates an Eigen::Map around existing memory in a 2D Kokkos::View with either LayoutLeft or LayoutRight layouts.
      
     @tparam ScalarType The scalar type stored in the Kokkos::View (e.g., double, unsigned int)
     @tparam OtherTraits Other properties of the Kokkos::View
     @param view The Kokkos matrix we wish wrap with an Eigen view.
     @return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::OuterStride<>>  And Eigen::Map with dynamic outer stride and contiguous inner stride.
     */
    template<typename ScalarType, typename... OtherTraits>
    inline Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**,OtherTraits...>>::Type, 0, Eigen::OuterStride<>> KokkosToMat(Kokkos::View<ScalarType**,OtherTraits...> const& view)
    {      
        size_t strides[2];
        view.stride(strides);
        assert((strides[0]==1)||(strides[1]==1));
        return Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::OuterStride<>>(view.data(), view.extent(0), view.extent(1), Eigen::OuterStride<>(std::max(strides[0],strides[1])));
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
    inline Eigen::Map<typename ViewToEigen<Kokkos::View<ScalarType**, OtherTraits...>>::Type, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>> KokkosToMat(Kokkos::View<ScalarType**, Kokkos::LayoutStride, OtherTraits...> const& view){
        
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
    inline typename ViewToEigen<ViewType>::Type CopyKokkosToMat(ViewType const& view){
        return KokkosToMat(view);
    }

}

#endif  // #ifndef ARRAYCONVERSIONS_H


