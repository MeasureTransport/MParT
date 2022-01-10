#ifndef MPART_MULTIINDEXSET_H
#define MPART_MULTIINDEXSET_H

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

namespace mpart{

class MultiIndexSet
{
public:

    /*
    Constructs a total order limited multiindex set
    */
    MultiIndexSet(unsigned int _dim, 
                  unsigned int _maxOrder);

    // Returns the maximum order in the dimension dim
    std::vector<unsigned int> GetMaxOrders() const;

    // Returns the multiindex with a given linear index
    std::vector<unsigned int> IndexToMulti(unsigned int index) const;

    // Returns the linear index of a given multiindex.  Returns -1 if not found.
    int MultiToIndex(std::vector<unsigned int> const& multi) const;

    void Print() const;

    unsigned int NumTerms() const;

    const unsigned int dim;

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

}; // class MultiIndexSet

} // namespace mpart


#endif