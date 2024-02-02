#ifndef MPART_MULTIINDEX_H_
#define MPART_MULTIINDEX_H_

#include <initializer_list>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace mpart {

class MultiIndexSet;


/**
@class MultiIndex
@brief A class for working with vectors of nonnnegative integers.
*/
class MultiIndex {
    friend class MultiIndexSet;

public:

    MultiIndex();

    /** Constructor that creates a multiindex with some default value.
        @param[in] lengthIn The length (i.e., number of components) in the multiindex.
        @param[in] val The value to be set for all entries.
    */
    MultiIndex(unsigned int lengthIn, unsigned int val=0);

    /** Takes a dense vector description of the multiindex and extracts the
        nonzero components.
        @param[in] indIn Row vector of unsigned integers defining the
                        multiindex.
    */
    MultiIndex(std::vector<unsigned int> const& indIn) : MultiIndex(&indIn[0], indIn.size()){};

    /** Constructs a multiindex from an Eigen vector. */
    template <typename Derived>
    MultiIndex(Eigen::MatrixBase<Derived> const& multi) : MultiIndex(multi.size(),0)
    {
        for(unsigned int i=0; i<length; ++i){
            if( multi(i) > 0 ){
                nzInds.push_back(i);
                nzVals.push_back(multi(i));
                maxValue = std::max<unsigned int>(maxValue, multi(i));
                totalOrder += multi(i);
            }
        }
    }

    /**
     * @brief Construct a new Multi Index object from a set of nonzero indices and values
     * 
     * @param nzIndsIn indices of the nonzero values
     * @param nzValsIn values that are nonzero
     * @param numNz number of nonzero values (i.e., length of nzIndsIn and nzValsIn)
     * @param lengthIn dimension of the index (numNz==lengthIn iff all values are nonzero)
     */
    MultiIndex(unsigned int* nzIndsIn,
        unsigned int* nzValsIn,
        unsigned int numNz,
        unsigned int lengthIn);
    
    /** Uses a dense vector description of the multiindex, defined through a pointer,
        and extracts the nonzero components.
        @param[in] fullVec A pointer the memory containing the dense multiindex.
        @param[in] length The length of the multiindex.
    */
    MultiIndex(const unsigned int* fullVec, unsigned int length);

    /** Allows users to intiailize the multiindex with curly braces.  For
        example, @code MultiIndex temp{1,0,2,3} @endcode would create a
        multiindex with length four and values 1, 0, 2, and 3.
    */
    MultiIndex(std::initializer_list<unsigned int> const& indIn);

    /** Get the dense representation of this multiindex.
        @return A std::vector of unsigned integers containing the multiindex.
    */
    std::vector<unsigned int> Vector() const;

    /** Get the total order of this multiindex: the \f$\ell_1\f$ norm.
    @return The sum of the nonzero components: the total order of this multiindex.
    */
    unsigned int Sum() const{return totalOrder;};

    /** This function returns the maximum degree of this multiindex: the \f$\ell_\infty\f$ norm.
    @return The maximum value of the multiindex.
    */
    unsigned int Max() const{return maxValue;};

    /** Use this function to set the value of the an entry in the multiindex.
    @param[in] ind The component of the multiindex to set (starting with 0).
    @param[in] val A non-negative value for the dim component of the multiindex.
    @return True if this function updated an already nonzero component, or false if this function added a new nonzero entry.
    */
    bool Set(unsigned int ind, unsigned int val);

    /** Obtain a particular component of the multiindex.  Notice that this function can be slow for multiindices with many nonzero components.  The worst case performance requires \f$O(|\mathbf{j}|_0)\f$ integer comparisons, where \f$|\mathbf{j}|_0\f$ denotes the number of nonzero entries in the multiindex.
    @param[in] ind The component to return.
    @return The integer stored in component dim of the multiindex.
    */
    unsigned int Get(unsigned int ind) const;

    /** Returns the number of nonzero components in the multiindex.
        @return An unsigned integer with the number of nonzero entries in the multiindex.
    */
    unsigned int NumNz() const;

    /** Return a string representation of the Multiindex.
        @return A string showing the dense representation of the multiindex, e.g., "[0,1,4,2]"
    */
    std::string String() const;

    /** Get the number of components in the index.  When used to define a
        multivariate polynomial, this will return the dimension of the
        polynomial.
        @return The length of the multiindex.  When used for defining multivariate polynomials,
                this will be the dimension of the polynomial.
    */
    unsigned int Length() const{return length;};

    /** Check for equality of the multiindices.
        @param[in] b The multiindex to compare with *this.
        @return true if both multiindices have the same lengths, nonzero indices, and nonzero values.  Returns false otherwise.
    */
    bool operator==(const MultiIndex &b) const;

    /** Check if two multiindices are different.
        @param[in] b The multiindex to compare with *this.
        @return false if both multiindices have the same lengths, nonzero indices, and nonzero values.  true otherwise.
    */
    bool operator!=(const MultiIndex &b) const;

    /**
        Check if this multiindex is less than b.  The multiindices are ordered such that:

        a<b if:
        - the length of a is less than the length of b OR
        - the lengths are the same, but the total order of "a" is less than the total order of "b" OR
        - the length and total orders are the same, but the max value of "a" is less than the max value of "b" OR
        - the lengths, total orders, and max values are the same, but "a" is lexicographically less than "b"

        @param[in] b The multiindex to compare with *this.
        @return true if *this < b according to the ordering described above.  false otherwise.
    */
    bool operator<(const MultiIndex &b) const;

    /**
        Check if this multiindex is greather than b.  The multiindices are ordered such that:

        a<b if:
        - the length of a is less than the length of b OR
        - the lengths are the same, but the total order of "a" is less than the total order of "b" OR
        - the length and total orders are the same, but the max value of "a" is less than the max value of "b" OR
        - the lengths, total orders, and max values are the same, but "a" is lexicographically less than "b"

        @param[in] b The multiindex to compare with *this.
        @return true if *this > b according to the ordering described above.  false otherwise.
    */
    bool operator>(const MultiIndex &b) const;

    /**
        Check if this multiindex is greater than or equal to b.  This is checked by returning
        "not (a<b)"

        @param[in] b The multiindex to compare with *this.
        @return true if *this >= b,  false otherwise.
    */
    bool operator>=(const MultiIndex &b) const;

    /**
        Check if this multiindex is greater than or equal to b.  This is checked by returning
        "not (a>b)"

        @param[in] b The multiindex to compare with *this.
        @return true if *this <= b,  false otherwise.
    */
    bool operator<=(const MultiIndex &b) const;

    /**
     * @brief Similar to operator>=, but bound must same length or longer. Further, it only
     * returns false if every value in this is less than every value in bound.
     *
     * @param bound bound for this multiindex
     * @return true if any value this[j] is at or above bound[j]
     * @return false else
     */
    bool AnyBounded(const MultiIndex &bound) const;

    /**
     * @brief Whether this index has a nonzero entry at the end
     * 
     * @return true if this has a nonzero entry at the end
    */
    bool HasNonzeroEnd() const;
private:

    unsigned int length;

    /// a vector holding pairs of (dimension,index) for nonzero values of index.
    std::vector<unsigned int> nzInds;
    std::vector<unsigned int> nzVals;

    /// The maximum index over all nzInds pairs.
    unsigned int maxValue;

    // the total order of the multiindex (i.e. the sum of the indices)
    unsigned int totalOrder;

}; // class MultiIndex


std::ostream& operator<< (std::ostream &out, mpart::MultiIndex const& ind);

} // namespace mpart


#endif
