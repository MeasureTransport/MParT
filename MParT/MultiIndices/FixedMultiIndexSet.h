#ifndef MPART_FIXEDMULTIINDEXSET_H
#define MPART_FIXEDMULTIINDEXSET_H

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

namespace mpart{

class FixedMultiIndexSet
{
public:

    /** @brief Construct a fixed multiindex set in dense form.
        
        All components of the multiindex are stored.  This requires more memory for high 
        dimensional problems, but might be easier to work with for some families of 
        basis functions.
    */
    FixedMultiIndexSet(unsigned int                _dim,
                       Kokkos::View<unsigned int*> _orders);

    /** @brief Construct a fixed multiindex set in compressed form.  
    
        Only nonzero orders are stored in this representation.   For multivariate polynomials,
        the compressed representation can yield faster polynomial evaluations.
    */
    FixedMultiIndexSet(unsigned int                _dim,
                       Kokkos::View<unsigned int*> _nzStarts,
                       Kokkos::View<unsigned int*> _nzDims,
                       Kokkos::View<unsigned int*> _nzOrders);
    /*
    Constructs a total order limited multiindex set
    */
    FixedMultiIndexSet(unsigned int _dim, 
                       unsigned int _maxOrder);

    // Returns the maximum degree in the dimension dim
    Kokkos::View<const unsigned int*> MaxDegrees() const;

    // Returns the multiindex with a given linear index
    std::vector<unsigned int> IndexToMulti(unsigned int index) const;

    // Returns the linear index of a given multiindex.  Returns -1 if not found.
    int MultiToIndex(std::vector<unsigned int> const& multi) const;

    void Print() const;

    unsigned int Size() const;

    const unsigned int dim;

    const bool isCompressed;

private:

    // Computes the number of terms in the multiindexset as well as the total number of nonzero components
    std::pair<unsigned int, unsigned int> TotalOrderSize(unsigned int maxOrder, unsigned int currDim);

    /**
     * @brief 
     * 
     * @param maxOrder The maximum total order (sum of powers) to include in multiindex set.
     * @param currDim The "top" dimension currently being filled in recursive calls to this function.  Starts at 0.
     * @param currNz The current index into the nzDims and nzOrders arrays. 
     */
    void FillTotalOrder(unsigned int maxOrder,
                        std::vector<unsigned int> &workspace, 
                        unsigned int currDim, 
                        unsigned int &currTerm,
                        unsigned int &currNz);


    Kokkos::View<unsigned int*> nzStarts;
    Kokkos::View<unsigned int*> nzDims;
    Kokkos::View<unsigned int*> nzOrders;
    Kokkos::View<unsigned int*> maxDegrees; // The maximum multiindex value (i.e., degree) in each dimension

}; // class MultiIndexSet

} // namespace mpart


#endif