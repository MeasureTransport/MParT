#ifndef MPART_MULTIINDEXSET_H_
#define MPART_MULTIINDEXSET_H_

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <iostream>

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"


namespace mpart{

class MultiIndexSet;

/** @class MultiIndexSet
  @ingroup MultiIndices
  @brief A class for holding, sorting, and adapting sets of multiindices.
  @details <p>In the context of polynomial expansions, a multiindex defines a
    single multivariate polynomial.  A finite expansion of multivariate
    polynomials is then defined by a collection of multiindices, one for
    each term in the expansion.  This class is a tool for defining such a
    multiindex set, relating members within the set (i.e. defining
    neighbors), and expanding the set. </p>

  <p>Let \f$\mbox{j}=[j_1,j_2,\dots,j_D]\f$ be a \f$D\f$-dimensional
    multiindex.  The backwards neighbors of \f$\mbox{j}\f$ are the multiindices
    given by multiindices who are only different from \f$\mbox{j}\f$ in one
    component, and in that component, the difference is -1.  For example,
    \f$[j_1-1, j_2,\dots,j_D]\f$ and \f$[j_1, j_2-1,\dots,j_D]\f$ are
    backwards neighbors of \f$\mbox{j}\f$, but \f$[j_1-1,
    j_2-1,\dots,j_D]\f$ and \f$[j_1, j_2-2,\dots,j_D]\f$ are not.  Forward
    neighbors are similarly defined, but with +1. Examples of forward
    neighbors include \f$[j_1+1, j_2,\dots,j_D]\f$ and \f$[j_1,
    j_2+1,\dots,j_D]\f$.   As far as this class is concerned, multiindices
    can be in three different categories: active, inactive, and/or
    admissable.  Active multiindices are those that are currently being used
    to define a polynomial expansion, inactive multiindices are not
    currently used but are being tracked, and admissable multiindices are
    inactive multiindices whose backward neighbors are all active.<p>

  <p>This class keeps track of both active and admissable multiindices, but
    only active indices are included in the linear indexing.  Nonactive
    indices are hidden (i.e. not even considered) in all the public members
    of this class.  For example, the GetAllMultiIndices function will return
    all active multiindices, and the IndexToMulti function will return
    \f$i^\mbox{th}\f$ active multiindex.  Inactive multiindices are used to
    check admissability and are added to the active set during adaptation.
  </p>

  <p>In general, members of the MultiIndexFactory class should be used to
    construct MultiIndexSets directly.<p>
*/
class MultiIndexSet{

  friend class MultiIndexFactory;

public:

  typedef std::function<bool(MultiIndex const&)> LimiterType;

  /**
   Factory method for constructing a total order limited multiindex set.
   @param[in] length The length of the multiindices stored in this set.
   @param[in] maxOrder The maximum order of multiindices to include.
   @param[in] limiter An optional additional limiter to attach to the set.  Only multiindices that satisfy this limiter will be included.
   @return MultiIndexSet An instance of the MultiIndexSet class containing all multiindices of order <= maxOrder AND satisfying the limiter.
  */
  static MultiIndexSet CreateTotalOrder(unsigned int length, 
                                        unsigned int maxOrder, 
                                        LimiterType const& limiter = MultiIndexLimiter::None());

  /**
   Factory method for constructing a full tensor product multiindex set.
   @param[in] length The length of the multiindices stored in this set.
   @param[in] maxDegree The maximum value allowed in any multiindex.
   @param[in] limiter An optional additional limiter to attach to the set.  Only multiindices that satisfy this limiter will be included.
   @return MultiIndexSet An instance of the MultiIndexSet class containing all multiindices with components a_i <= maxDegree AND satisfying the limiter.
  */
  static MultiIndexSet CreateTensorProduct(unsigned int length, 
                                           unsigned int maxOrder, 
                                           LimiterType const& limiter = MultiIndexLimiter::None());


  /**
   @brief Construct a new MultiIndexSet object with a specific length.
   @details Constructs an empty MultiIndexSet to store multi-indices of a specified length.  Also allows 
            a functor to be be passed in as an additional limiter on the admissible set.  If no functor is 
            provided, it is possible to include any multi-index in \f$\mathbb{N}^D\f$.  

            For example, to construct a MultiIndexSet in 2 dimensions that only allows multi-indices with maximum 
            degree less than 5, we could use a lambda function:
@code{.cpp}
unsigned int length =2;
auto limiter = [](MultiIndex const& multi) {return multi.Max()<5;};
MultiIndexSet set(length, limiter);
@endcode
            Multiple pre-defined limiter functors are defined in the mpart::MultiIndexLimiter namespace.

   @param[in] lengthIn The length of each multi-index in the set.
   @param[in] limiterIn An optional functor defining the possible admissible multi-indices.  Should accept a const& to MultiIndex and return a boolean.  True if the MultiIndex is allowed and false otherwise.
  */
  MultiIndexSet(const unsigned int lengthIn,
                LimiterType const& limiterIn = MultiIndexLimiter::None() );

  /**
   @brief "Compresses" this multiindex set into the fixed representation provided by the "FixedMultiIndexSet" class.
   @details The FixedMultiIndexSet cannot easily be adapted, but stores the multiindices in a contiguous block of memory 
            in a Kokkos::View that can be more amenable to fast computation.   This function creates an instance of the 
            FixedMultiIndexSet from the current state of *this.   Note that memory is deep copied and any subsequent updates 
            to this class will not result in updates to the FixedMultiIndexSet.
   @return An instance of the FixedMultiIndexSet class with a snapshot of the current state of this MultiIndexSet.
   */
  FixedMultiIndexSet Compress() const;

  /** Set the limiter of this MultiIndexSet.  This function will check to make
      sure that all currently active nodes are still feasible with the new limiter.
      If this is not the case, an assert will be thrown.
      @param[in] limiterIn A functor that accepts a MultiIndex and returns a boolean if it is allowed in the set.
  */
  void SetLimiter(LimiterType const& limiterIn);

  /** Returns the limiter used in this MultiIndexSet. */
  LimiterType GetLimiter() const{return limiter;};

  /** Given an index into the set, return the corresponding multiindex as an
      instance of the MultiIndex set. If all the multiindices were stored in a
      vector called multiVec, the functionality of this method would be equivalent
      to multiVec[activeIndex].
      @param[in] activeIndex Linear index of interest.
      @return A constant reference to the MultiIndex.
  */
  MultiIndex const& IndexToMulti(unsigned int activeIndex) const{return allMultis.at(active2global.at(activeIndex));};

  /** Given a multiindex, return the linear index where it is located.
      @param[in] input An instance of the MultiIndex class.
      @return If the multiindex was found in this set, a nonnegative value
      containing the linear index is returned.  However, if the set does not
      contain the multiindex, -1 is returned.
  */
  int MultiToIndex(MultiIndex const& input) const;

  /** Get the dimension of the multiindex, i.e. how many components does it have? */
  unsigned int Length() const{return length;};

  /** Assume the \f$\mathbf{j}^{\mbox{th}}\f$ multiindex in this set is given by
      \f$\mathbf{j}=[j_1,j_2,\dots,j_D]\f$.  This function returns a vector
      containing the maximum value of any multiindex in each direction, i.e., a
      vector \f$\mathbf{m}=[m_1,m_2,\dots,m_D]\f$ where \f$m_d = \max_{\mathbf{j}}
      j_d\f$.
      @return The vector \f$\mathbf{m}\f$.
  */
  std::vector<unsigned int> const& MaxOrders() const{return maxOrders;};

  /**
    * This function provides access to each of the MultiIndices.
    @param[in] activeIndex The index of the active MultiIndex to return.
    @return A pointer to the MultiIndex at index outputIndex.
    */
  MultiIndex const& at(int activeIndex) const{return IndexToMulti(activeIndex);}

  /**
    * This function provides access to each of the MultiIndices without any bounds checking on the vector.
    @param[in] outputIndex The index of the active MultiIndex we want to return.
    @return A pointer to the MultiIndex at index outputIndex.
    */
  MultiIndex const& operator[](int activeIndex) const{return allMultis[active2global[activeIndex]]; };

  /**
    * Get the number of active MultiIndices in this set.
    @return An unsigned integer with the number of active MultiIndices in the set.
    */
  unsigned int Size() const{return active2global.size();};

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * * * ADAPTIVE COMPONENTS
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  /** @brief Add another set of multiindices to this one.
      @details Any multiindices the rhs MultiIndexSet not already in (*this) are added.  After calling this function, *this will contain the union or the two sets.
      @param[in] rhs Another MultiIndex set to add to this one
      @return A reference to this MultiIndex set, which now contains the union of
              this set and rhs.
  */
  MultiIndexSet& operator+=(const MultiIndexSet &rhs);

  /** @brief Add a single MultiIndex to the set.
    @details This functions checks to see if the input basis function is
            already in the set and if the input function is unique, it is
            added to the set.
    @param[in] rhs The MultiIndex we want to add to the set.
    @return A reference to this MultiIndex set, which may now contain the new
            MultiIndex in rhs.
    */
  MultiIndexSet& operator+=(MultiIndex const& rhs);

  /** @brief Add all terms in rhs to this instance.
    @details This function adds all unique MultiIndices from the rhs into this MultiIndexSet.  In the event that a multiindex is active in one set, but not the other, the union will set that multiindex to be active.
    @param[in] rhs The MultiIndex set we want to add to this instance.
    @return The number of unique terms from rhs that were added to this set.
    */
  unsigned int Union(const MultiIndexSet &rhs);

  /**
      Make the multi-index active. If the multiIndex is not admissable, an exception 
      will be thrown.  To be admissable (according to this function), the multiIndex
      must already exist as an inactive member of this set.  If that is not the case,
      use the AddActive function instead.
      @param[in] multiIndex A multiindex to make active. Note that an assert will fail if multiIndex is not admissable.
  */
  void Activate(MultiIndex const& multiIndex);

  /**
    * Add the given multiindex to the set and make it active.  The functionality
      of this function is very similar to Activate; however, this function will
      add the multiIndex if it does not already exist in the set.  This function
      does not check for admissability.  Instead, it will add the multiindex to
      the set and add all neighbors as inactive.  Be careful when using this
      function as it is possible to create a set with active multiindices that
      are not admisable.
    @param[in] newNode A multiindex we want to add as an active member of the set.
    @return An integer specifying the linear index of the now active multiindex.
    */
  int AddActive(MultiIndex const& newNode);

  /**
      If possible, make the neighbors of this index active, and return any
      that become active.  Do not activate a neighbor that is already part
      of the family.
      @param activeIndex The linear index of the active multiindex to expand.
      @return A vector containing the linear index of any multiindices
              activated because of the expansion.
    */
  std::vector<unsigned int> Expand(unsigned int activeIndex);

   /**
      Activate any inactive but admissible forward neighbors of MultiIndices on the frontier.
      @return A vector containing the linear index of any multiindices
              activated because of this expansion.
    */
   std::vector<unsigned int> Expand();

  /**
      Completely expands an index, whether or not it is currently expandable.
      In order to maintain admissability of the set, it will add backward
      neighbors needed recursively, and return a list of all the indices it adds.
      @param activeIndex The linear index of the active multiindex to expand.
      @return A vector containing the linear index of any multiindices
              activated because of the expansion.
    */
  std::vector<unsigned int> ForciblyExpand(unsigned int const activeIndex);

  /**
      Add the given multi-index to the active set regardless of whether it's currently admissible.
      To keep the whole set admissible, recursively add any backward neighbors necessary.
      Returns a list of indices of any newly added elements, including itself.
      @param multiIndex The MultiIndex to forcibly add, make active, and make admissable.
      @return A vector of linear indices indicating all the active MultiIndices
              added to the set in order to make the given multiIndex admissable.
    */
  std::vector<unsigned int> ForciblyActivate(MultiIndex const& multiIndex);

  /** This function returns the admissable forward neighbors of an active multiindex.
    @param[in] activeIndex The linear index of the active multiIndex under consideration.
    @return A vector of admissible forward neighbors.
    */
  std::vector<MultiIndex> GetAdmissibleForwardNeighbors(unsigned int activeIndex);

  /** Here, we define a term on the "frontier" of the multiindex set as one
      that has at least one inactive admissable forward neighbors.  These terms are expandable.
      @return A vector with linear indices for active multi-indices on the frontier.
  */
  std::vector<unsigned int> GetFrontier() const;

  /** We define the strict frontier to be the collection of multiindices, whose
      forward neighbors are all inactive.
      @return A vector with linear indices for active multi-indices on the strict frontier.
  */
  std::vector<unsigned int> GetStrictFrontier() const;

  /** Returns the indices for the backward neighbors of a currently active multiindex.
  @param[in] activeIndex The linear index of the MultiIndex of interest
  @return A std::vector containing the linear indices of the backward neighbors.
  */
  std::vector<unsigned int> GetBackwardNeighbors(unsigned int activeIndex) const;

  /** Returns indices for backward neighbors of an active or inactive multiindex. 
  @param[in] multiIndex The multiindex in question.
  @return A vector with linear indices for the backward neighbors of this multiindex.
  */
  std::vector<unsigned int> GetBackwardNeighbors(MultiIndex const& multiIndex) const;

  /**
  Determines whether the input multiIndex is currently admissible.
  @param[in] multiIndex The multiindex to consider.
  @return true if the multiIndex is admissible.  false otherwise.
  */
  bool IsAdmissible(MultiIndex const& multiIndex) const;

  /** 
  Check to see if any forward neighbors of a multiIndex are admissible but not active.
  @param[in] activeIndex The linear index of the multi-index in question.
  @return true if this multiindex has at least one admissible but inactive forward neighbor.
  */
  bool IsExpandable(unsigned int activeIndex) const;

  ///Return true if the multiIndex is active
  bool IsActive(MultiIndex const& multiIndex) const;

  /// Returns the number of active forward neighbors
  unsigned int NumActiveForward(unsigned int activeInd) const;

  /// Returns the number of forward neighbors (active or inactive)
  unsigned int NumForward(unsigned int activeInd) const;

  /** Visualizes a two-dimensional MultiIndexSet as ASCII art.   The output for a total order limited set with max order 11 looks like 
  @code
12 | o  
11 | x  o  
10 | x  x  o  
 9 | x  x  x  o  
 8 | x  x  x  x  o  
 7 | x  x  x  x  x  o  
 6 | x  x  x  x  x  x  o  
 5 | x  x  x  x  x  x  x  o  
 4 | x  x  x  x  x  x  x  x  o  
 3 | x  x  x  x  x  x  x  x  x  o  
 2 | x  x  x  x  x  x  x  x  x  x  o  
 1 | x  x  x  x  x  x  x  x  x  x  x  o  
 0 | x  x  x  x  x  x  x  x  x  x  x  x  o  
    ----------------------------------------
     0  1  2  3  4  5  6  7  8  9  10 11 12 
  @endcode 

  @param[in,out] out The output stream where the visualization should be written.  Defaults to std::cout.  
  */ 
  void Visualize(std::ostream &out = std::cout) const;

protected:

  // A vector of both active and admissable multiindices.  Global index.
  std::vector<MultiIndex> allMultis;

  // store a MultiIndexLimiter that will tell us the feasible set (i.e. simplex, exp limited, etc...)
  LimiterType limiter;

  // the dimension (i.e., number of components) in each multi index
  const unsigned int length;

  int AddInactive(MultiIndex const& newNode);

  virtual bool IsAdmissible(unsigned int globalIndex) const;
  virtual bool IsActive(unsigned int globalIndex) const;

  void AddForwardNeighbors(unsigned int globalIndex, bool addInactive);
  void AddBackwardNeighbors(unsigned int globalIndex, bool addInactive);

  void Activate(int globalIndex);
  void ForciblyActivate(int localIndex, std::vector<unsigned int> &newInds);

  // Maps the active index to an entry in allMultis
  std::vector<unsigned int> active2global;

  // Maps a global index to an active index.  Non-active values are -1
  std::vector<int> global2active;

  // a vector of sets for the input and output edges.
  std::vector<std::set<int>> outEdges; // edges going out of each multi
  std::vector<std::set<int>> inEdges;  // edges coming in to each multi

  std::vector<unsigned int> maxOrders; // the maximum order in each dimension

private:

  int AddMulti(MultiIndex const& newMulti);

  static void RecursiveTotalOrderFill(unsigned int   maxOrder, 
                                      MultiIndexSet &mset,
                                      unsigned int currDim,
                                      std::vector<unsigned int> &denseMulti,
                                      LimiterType const& limiter);

  static void RecursiveTensorFill(unsigned int   maxOrder, 
                                      MultiIndexSet &mset,
                                      unsigned int currDim,
                                      std::vector<unsigned int> &denseMulti,
                                      LimiterType const& limiter);

  std::map<MultiIndex, unsigned int> multi2global; // map from a multiindex to an integer

}; // class MultiIndexSet

} // namespace mpart




#endif // MPART_MULTIINDEXSET_H_
