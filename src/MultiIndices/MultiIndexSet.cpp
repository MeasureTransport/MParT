#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "MParT/MultiIndices/MultiIndexSet.h"

using namespace mpart;

MultiIndexSet MultiIndexSet::CreateTotalOrder(unsigned int length,
                                              unsigned int maxOrder,
                                              LimiterType const& limiter)
{
    assert(length>0);

    // create an empy multiindex set
    MultiIndexSet output(length, limiter);

    // start with a vector of zeros
    std::vector<unsigned int> base(length,0);

    RecursiveTotalOrderFill(maxOrder, output, 0, base, limiter);

    return output;
}


MultiIndexSet MultiIndexSet::CreateTensorProduct(unsigned int length,
                                                 unsigned int maxDegree,
                                                 LimiterType const& limiter)
{
    assert(length>0);

    // create an empy multiindex set
    MultiIndexSet output(length, limiter);

    // start with a vector of zeros
    std::vector<unsigned int> base(length,0);

    RecursiveTensorFill(maxDegree, output, 0, base, limiter);

    return output;
}

void MultiIndexSet::RecursiveTotalOrderFill(unsigned int   maxOrder,
                                            MultiIndexSet &output,
                                            unsigned int currDim,
                                            std::vector<unsigned int> &base,
                                            LimiterType const& limiter)
{
    unsigned int currOrder = 0;
    for(unsigned int i=0; i<currDim; ++i)
        currOrder += base.at(i);

    const int length = base.size();

    if(currDim==length-1)
    {
        for(int i=0; i<=maxOrder-currOrder; ++i)
        {
            base.at(length-1) = i;
            MultiIndex newTerm(base);
            if(limiter(newTerm))
                output.AddActive(newTerm);
        }

    }else{
        for(int i=0; i<=maxOrder-currOrder; ++i)
        {
            for(unsigned int k=currDim+1; k<length; ++k)
                base.at(k) = 0;

            base.at(currDim) = i;
            RecursiveTotalOrderFill(maxOrder,output,currDim+1,base,limiter);
        }
    }
}


void MultiIndexSet::RecursiveTensorFill(unsigned int   maxDegree,
                                        MultiIndexSet &output,
                                        unsigned int currDim,
                                        std::vector<unsigned int> &base,
                                        LimiterType const& limiter)
{
    const int length = base.size();

    if(currDim==length-1)
    {
        for(int i=0; i<=maxDegree; ++i)
        {
            base.at(length-1) = i;
            MultiIndex newTerm(base);
            if(limiter(newTerm))
                output.AddActive(newTerm);
        }

    }else{
        for(int i=0; i<=maxDegree; ++i)
        {
            for(unsigned int k=currDim+1; k<length; ++k)
                base.at(k) = 0;

            base.at(currDim) = i;
            RecursiveTensorFill(maxDegree,output,currDim+1,base,limiter);
        }
    }
}



MultiIndexSet::MultiIndexSet(const unsigned int lengthIn,
                             LimiterType const& limiterIn,
                             std::shared_ptr<MultiIndexNeighborhood> neigh) : maxOrders(lengthIn,0),
                                                                              length(lengthIn),
                                                                              limiter(limiterIn),
                                                                              neighborhood(neigh)
{
};


FixedMultiIndexSet MultiIndexSet::Fix(bool compress) const
{
  if(compress){

    unsigned int numTerms = Size();
    unsigned int totalNnz = 0; // total number of nonzero components in all multiindex
    for(auto& activeInd : active2global)
      totalNnz += allMultis.at(activeInd).NumNz();


    Kokkos::View<unsigned int*> nzStarts("Start of a Multiindex", numTerms+1);
    Kokkos::View<unsigned int*> nzDims("Index of nz component", totalNnz);
    Kokkos::View<unsigned int*> nzOrders("Power of nz component", totalNnz);

    unsigned int cumNz = 0;

    for(unsigned int i=0; i<numTerms; ++i){

      unsigned int activeInd = active2global.at(i);
      MultiIndex const& multi = allMultis.at(activeInd);

      nzStarts(i) = cumNz;

      for(unsigned int j=0; j<multi.nzInds.size(); ++j){
        nzDims(cumNz + j) = multi.nzInds[j];
        nzOrders(cumNz + j) = multi.nzVals[j];
      }

      cumNz += allMultis.at(activeInd).NumNz();
    }
    nzStarts(numTerms) = totalNnz;

    return FixedMultiIndexSet(length, nzStarts, nzDims, nzOrders);

  }else{

    Kokkos::View<unsigned int*> orders("orders", length*Size());
    std::vector<unsigned int> multi;

    for(unsigned int i=0; i<Size(); ++i){
      multi = IndexToMulti(i).Vector();

      for(unsigned int d=0; d<length; ++d)
        orders(d + i*length) = multi.at(d);

    }

    return FixedMultiIndexSet(length, orders);
  }
}


void MultiIndexSet::SetLimiter(LimiterType const& newLimiter){

  // copy the limiter
  limiter = newLimiter;

  //  make sure no active terms in the set currently obey the new limiter.
  //  If a term is inactive, remove all edges tied to it
  for(int globalInd=0; globalInd<allMultis.size(); ++globalInd)
  {
    if(IsActive(globalInd)){
      if(!newLimiter(allMultis.at(globalInd))){
        std::stringstream msg;
        msg << "Invalid limiter passed to MultiIndexSet::SetLimiter.  The active multi-index, ";
        msg << allMultis.at(globalInd).Vector() << ", is not valid with the new limiter.\n";
        throw std::invalid_argument(msg.str());
      }

      // Add any newly admissible inactive forward neighbors
      AddForwardNeighbors(globalInd,true);

    }else{

      if(!newLimiter(allMultis.at(globalInd))){
        for(int inNode : inEdges[globalInd])
          outEdges[inNode].erase(globalInd);
        inEdges[globalInd].clear();
      }
    }
  }


}

int MultiIndexSet::MultiToIndex(MultiIndex const& input) const{

  auto localIter = multi2global.find(input);

  if(localIter!=multi2global.end()){
    return global2active[localIter->second];
  }else{
    return -1;
  }
}


int MultiIndexSet::AddMulti(MultiIndex const& newMulti)
{
  allMultis.push_back(newMulti);

  int globalInd = allMultis.size() - 1;
  multi2global[allMultis.back()] = globalInd;

  global2active.push_back(-1);

  inEdges.push_back(std::set<int>());
  outEdges.push_back(std::set<int>());

  assert(allMultis.size() == global2active.size());

  AddForwardNeighbors(globalInd,false);
  AddBackwardNeighbors(globalInd, false);

  return globalInd;
}

int MultiIndexSet::AddActive(MultiIndex const& newNode)
{
  int globalInd = AddInactive(newNode);

  if(globalInd>=0){

    Activate(globalInd);
    return global2active[globalInd];

  }else{
    return -1;
  }
}



int MultiIndexSet::AddInactive(MultiIndex const& newNode)
{
  auto iter = multi2global.find(newNode);

  if(iter!=multi2global.end()){
    return iter->second;

  }else if(limiter(newNode)){
    return AddMulti(newNode);

  }else{
    return -1;
  }
}

bool MultiIndexSet::IsActive(MultiIndex const& multiIndex) const
{
  auto iter = multi2global.find(multiIndex);

  if(iter!=multi2global.end()){
    return IsActive(iter->second);
  }else{
    return false;
  }
}

bool MultiIndexSet::IsActive(unsigned int globalIndex) const
{
  return global2active[globalIndex] >= 0;
}

bool MultiIndexSet::IsAdmissible(unsigned int globalIndex) const
{
  auto& multi = allMultis.at(globalIndex);

  if(!limiter(multi))
    return false;

  if(IsActive(globalIndex))
    return true;

  // count the number of input edges that are coming from active indices
  int numAdmiss = 0;
  for(int inNode : inEdges.at(globalIndex)){
    if(IsActive(inNode))
      numAdmiss++;
  }

  if(numAdmiss==multi.NumNz()){
    return true;
  }else{
    return false;
  }
}

bool MultiIndexSet::IsAdmissible(MultiIndex const& multiIndex) const
{
  auto iter = multi2global.find(multiIndex);

  if(iter==multi2global.end()){
    return false;
  }else{
    return IsAdmissible(iter->second);
  }
}


bool MultiIndexSet::IsExpandable(unsigned int activeIndex) const
{
  // an index is expandable when at least one forward neighbor is admissible but not active (i.e. outedge)

  // loop through the outgoing edges for this node
  for(int nextInd : outEdges[active2global.at(activeIndex)]){
    if(!IsActive(nextInd)&&IsAdmissible(nextInd))
      return true;
  }
  return false;
}

void MultiIndexSet::Activate(int globalIndex)
{

  // the index is already in the global set, if the value is non-negative, it is also active and we don't need to do anything
  if(global2active.at(globalIndex)<0)
  {
    auto& newNode = allMultis.at(globalIndex);

    // now add the index to the active set
    active2global.push_back(globalIndex);

    int newActiveInd = active2global.size()-1;

    global2active.at(globalIndex) = newActiveInd;

    // update the maximum order
    for(unsigned int i=0; i<newNode.nzInds.size(); ++i)
      maxOrders.at(newNode.nzInds.at(i)) = std::max<unsigned int>(maxOrders.at(newNode.nzInds.at(i)),newNode.nzVals.at(i));

    AddForwardNeighbors(globalIndex,true);
    AddBackwardNeighbors(globalIndex,true);
  }
}

void MultiIndexSet::Activate(MultiIndex const& multiIndex)
{
  auto iter = multi2global.find(multiIndex);

  assert(iter!=multi2global.end());
  assert(IsAdmissible(iter->second));

  Activate(iter->second);
}

void MultiIndexSet::AddForwardNeighbors(unsigned int globalIndex, bool addInactive)
{
  std::vector<MultiIndex> neighbors = neighborhood->ForwardNeighbors(allMultis.at(globalIndex));

  for(auto& multi : neighbors)
  {

    // If this is within the limiter set
    if(limiter(multi)){

      // Check to see if we already have this multiindex...
      auto iter = multi2global.find(multi);
      if(iter!=multi2global.end()){
        inEdges.at(iter->second).insert(globalIndex);
        outEdges.at(globalIndex).insert(iter->second);

      // If not, add it
      }else if(addInactive){
        AddInactive(multi);
      }
    }

  }
}


void MultiIndexSet::Visualize(std::ostream &out) const
{

  for(int i=maxOrders.at(1)+1; i>=0; --i){

    if(i<10)
      out << " ";
    out << i << " | ";

    for(unsigned int j=0; j<=maxOrders.at(0)+1; ++j){

      bool found = false;
      for(unsigned int k=0; k<active2global.size(); ++k){
        if((allMultis.at(active2global.at(k)).Get(0)==j)&&(allMultis.at(active2global.at(k)).Get(1)==i)){
          out << "a  ";
          found = true;
          break;
        }
      }

      if(!found){
        for(auto& multi : allMultis){
          if((multi.Get(0)==j)&&(multi.Get(1)==i)){
            if(IsAdmissible(multi)){
              out << "r  ";
            }else{
              out << "m  ";
            }
          }
        }
      }
    }
    out << std::endl;
  }

  out << "    -";
  for(unsigned int j=0; j<=maxOrders.at(0)+1; ++j)
    out << "---";

  out << "\n     ";
  for(unsigned int j=0; j<=maxOrders.at(0)+1; ++j){

    if(j<10)
      out << j << "  ";
    else
      out << j << " ";
  }
  out << std::endl;

}


std::vector<MultiIndex>  MultiIndexSet::AdmissibleForwardNeighbors(unsigned int activeIndex)
{
  unsigned int globalInd = active2global.at(activeIndex);

  std::vector<MultiIndex> output;
  for( auto neighbor : outEdges[globalInd])
  {
    if(IsAdmissible(neighbor))
      output.push_back(allMultis.at(neighbor));
  }

  return output;
}

std::vector<unsigned int> MultiIndexSet::Frontier() const {

  std::vector<unsigned int> frontierInds;

  for(unsigned int activeInd = 0; activeInd<active2global.size(); ++activeInd) {
    if(IsExpandable(activeInd))
      frontierInds.push_back(activeInd);
  }

  return frontierInds;
}

std::vector<MultiIndex> MultiIndexSet::Margin() const
{
  std::vector<MultiIndex> output;

  for(unsigned int globalInd=0; globalInd<global2active.size(); ++globalInd){

    // If this is an inactive multiindex
    if(!IsActive(globalInd)){

      // Check the backward neighbors
      for(auto neighbor : inEdges[globalInd]){
        if(IsActive(neighbor)>0){
          output.push_back(allMultis.at(globalInd));
          break;
        }
      }
    }
  }

  return output;
}


std::vector<MultiIndex> MultiIndexSet::ReducedMargin() const
{

  std::vector<MultiIndex> output;

  for(unsigned int globalInd=0; globalInd<global2active.size(); ++globalInd){

    // If this is an inactive multiindex
    if(!IsActive(globalInd)){

      // Check the backward neighbors
      bool allActive = true;
      for(auto neighbor : inEdges[globalInd])
        allActive = (allActive && IsActive(neighbor));

      if(allActive)
        output.push_back(allMultis.at(globalInd));

    }
  }

  return output;
}


std::vector<unsigned int> MultiIndexSet::StrictFrontier() const
{
  std::vector<unsigned int> frontInds = Frontier();
  std::vector<unsigned int> strictInds;

  for(unsigned int i=0; i<frontInds.size(); ++i) {

    unsigned int activeInd = frontInds.at(i);
    unsigned int globalInd = active2global.at(activeInd);

    // Check to make sure all forward neighbors are inactive
    bool isStrict = true;
    for( auto neighbor : outEdges[globalInd]){
      if(IsActive(neighbor)){
        isStrict = false;
        break;
      }
    }

    if(isStrict)
      strictInds.push_back(activeInd);
  }

  return strictInds;
}

std::vector<unsigned int> MultiIndexSet::BackwardNeighbors(unsigned int activeIndex) const
{
  unsigned int globalInd = active2global.at(activeIndex);

  std::vector<unsigned int> output;
  for(auto neighbor : inEdges[globalInd])
    output.push_back(global2active.at(neighbor));

  return output;
}

std::vector<unsigned int> MultiIndexSet::BackwardNeighbors(MultiIndex const& multiIndex) const
{
  auto iter = multi2global.find(multiIndex);

  assert(iter!=multi2global.end());

  unsigned int globalInd = iter->second;
  std::vector<unsigned int> output;
  for(auto neighbor : inEdges[globalInd])
    output.push_back(global2active.at(neighbor));

  return output;
}


unsigned int MultiIndexSet::NumActiveForward(unsigned int activeInd) const
{
  unsigned int globalInd = active2global.at(activeInd);

  unsigned int numActive = 0;
  for( auto neighbor : outEdges[globalInd])
  {
    if(IsActive(neighbor))
      numActive++;
  }
  return numActive;
}

unsigned int MultiIndexSet::NumForward(unsigned int activeInd) const
{
  unsigned int globalInd = active2global.at(activeInd);
  return outEdges[globalInd].size();
}

void MultiIndexSet::AddBackwardNeighbors(unsigned int globalIndex, bool addInactive)
{
  std::vector<MultiIndex> neighbors = neighborhood->BackwardNeighbors(allMultis.at(globalIndex));

  for(auto& multi : neighbors)
  {
    if(limiter(multi)){

      // Check to see if we already have this multiindex in the set
      auto iter = multi2global.find(multi);
      if(iter!=multi2global.end()){
        outEdges.at(iter->second).insert(globalIndex);
        inEdges.at(globalIndex).insert(iter->second);

      // If not, add it
      }else if(addInactive){
        AddInactive(multi);
      }
    }

  }
}

std::vector<unsigned int> MultiIndexSet::Expand(unsigned int activeIndex)
{
  if(activeIndex >= active2global.size()){
    std::stringstream msg;
    msg << "Invalid index passed to MultiIndexSet::Expand.  A value of " << activeIndex << " was passed to the function, but only " << active2global.size() << " active components exist in the set.\n";
    throw std::out_of_range(msg.str());
  }

  std::vector<unsigned int> newIndices;
  unsigned int globalIndex = active2global.at(activeIndex);

  // loop through the forward neighbors of this index
  std::set<int> tempSet = outEdges.at(globalIndex);
  for(int neighbor : tempSet)
  {
    if(IsAdmissible(neighbor)&&(!IsActive(neighbor))){
      Activate(neighbor);
      newIndices.push_back(global2active.at(neighbor));
    }
  }

  // return the vector of newly activated indices
  return newIndices;
}

std::vector<unsigned int> MultiIndexSet::ForciblyExpand(unsigned int const activeIndex)
{
  assert(activeIndex<active2global.size());

  std::vector<unsigned int> newIndices;
  unsigned int globalIndex = active2global.at(activeIndex);

  // loop through the forward neighbors of this index
  std::set<int>& tempSet = outEdges.at(globalIndex);
  for(int neighbor : tempSet)
    ForciblyActivate(neighbor,newIndices);

  // return the vector of newly activated indices
  return newIndices;

}

void MultiIndexSet::ForciblyActivate(int globalIndex, std::vector<unsigned int> &newIndices){


  if(!IsActive(globalIndex)){

    // make the node active and add inactive neighbors if necessary, this also updates the edges and enables the loop below
    Activate(globalIndex);
    newIndices.push_back(global2active.at(globalIndex));

    // now, fill in all of the previous neighbors
    std::set<int>& tempSet = inEdges.at(globalIndex);
    for(int ind : tempSet)
      ForciblyActivate(ind,newIndices);

  }
}

std::vector<unsigned int> MultiIndexSet::ForciblyActivate(MultiIndex const& multiIndex){

  assert(limiter(multiIndex));

  auto iter = multi2global.find(multiIndex);
  std::vector<unsigned int> newIndices;

  // if we found the multiindex and it is active, there is nothing to do
  if(iter!=multi2global.end()){
    ForciblyActivate(iter->second,newIndices);
  }else{
    // Add the new index as an active node
    int newGlobalInd = AddInactive(multiIndex);
    ForciblyActivate(newGlobalInd,newIndices);
  }

  return newIndices;
}

MultiIndexSet& MultiIndexSet::operator+=(const MultiIndexSet& rhs)
{
  Union(rhs);
  return *this;
}

unsigned int MultiIndexSet::Union(const MultiIndexSet& rhs)
{
  int oldTerms = Size();

  for(int i = 0; i < rhs.allMultis.size(); ++i) {

    auto newMulti = rhs.allMultis.at(i);
    if(limiter(newMulti)){
      if(rhs.global2active[i]<0){
        AddInactive(newMulti);
      }else{
        AddActive(newMulti);
      }
    }
  }

  return Size() - oldTerms;
}

MultiIndexSet& MultiIndexSet::operator+=(MultiIndex const& rhs)
{
  AddActive(rhs);
  return *this;
}
