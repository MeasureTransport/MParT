#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>

using namespace mpart;

MultiIndex::MultiIndex() : MultiIndex((unsigned int)0, (unsigned int)0) {};

MultiIndex::MultiIndex(unsigned int lengthIn, unsigned val) : length(lengthIn),
                                                                maxValue(val),
                                                                totalOrder(lengthIn*val)
{
  if(val>0){
      nzVals.resize(length,val);

      nzInds.resize(length);
      for(unsigned int i=0; i<length; ++i)
        nzInds[i] = i;
  }
}


MultiIndex::MultiIndex(const unsigned int* fullVec,
                       unsigned int lengthIn) : MultiIndex(lengthIn, 0)
{
    for(unsigned int i=0; i<length; ++i){
        if( fullVec[i] > 0 ){
            nzInds.push_back(i);
            nzVals.push_back(fullVec[i]);
            maxValue = std::max<unsigned int>(maxValue, fullVec[i]);
            totalOrder += fullVec[i];
        }
    }
}


MultiIndex::MultiIndex(std::initializer_list<unsigned int> const& indIn) : MultiIndex(indIn.size(),0)
{
  maxValue = 0;
  totalOrder = 0;

  unsigned int i = 0;
  for(auto it = indIn.begin(); it != indIn.end(); ++it){
    if( *it > 0 ){
      nzInds.push_back(i);
      nzVals.push_back(*it);

      maxValue = std::max<int>(maxValue, *it);
      totalOrder += *it;
    }
    i++;
  }
}

MultiIndex::MultiIndex(unsigned int* nzIndsIn,
                       unsigned int* nzValsIn,
                       unsigned int numNz,
                       unsigned int lengthIn) : length(lengthIn),
                                                maxValue(0),
                                                totalOrder(0)
{
  for(unsigned int i=0; i<numNz; ++i){
    if(nzValsIn[i]>0){
      nzInds.push_back(nzIndsIn[i]);
      nzVals.push_back(nzValsIn[i]);
      maxValue = std::max<unsigned int>(maxValue, nzValsIn[i]);
      totalOrder += nzValsIn[i];
    }
  }
}

std::vector<unsigned int>MultiIndex::Vector() const
{
  std::vector<unsigned int> output(length,0);

  for(unsigned int i=0; i<nzVals.size(); ++i)
      output[nzInds[i]] = nzVals[i];

  return output;
}

bool MultiIndex::Set(unsigned int ind, unsigned int val)
{
  if(ind>length){
    throw std::out_of_range("Tried to set the value of index " + std::to_string(ind) + " on an multiindex with only " + std::to_string(length) + " components.");
    return false;
  }else{

    bool existingNz = false;

    // Get an iterator into the nzInds that matches ind, or is where we want to insert a new value
    auto indIter = std::lower_bound(nzInds.begin(), nzInds.end(), ind);
    unsigned int index = std::distance(nzInds.begin(), indIter);

    // Check to see if we already have a nonzero value at this index
    if(indIter!=nzInds.end()){
        if((*indIter)==ind){
            existingNz = true;
        }
    }

    if(existingNz){

        // If the new value is nonzero, update the value
        if(val>0){
            existingNz = true;
            nzVals.at(index) = val;

        // If the new value is zero, erase this entry
        }else{
            nzInds.erase(indIter);
            nzVals.erase(nzVals.begin()+index);
        }

    }else if(val==0){
        // We don't already have this index, and the new val is just zero, so we don't need to change anything
        return false;

    // This is a new nonzero component that we need to insert
    }else{
        nzInds.insert(indIter, ind);
        nzVals.insert(nzVals.begin()+index, val);
    }

    // Update the total and maximum order values after updating multi index
    totalOrder = 0;
    maxValue = 0;

    for(unsigned int i=0; i<nzVals.size(); ++i){
      totalOrder += nzVals.at(i);
      maxValue = std::max(maxValue, nzVals.at(i));
    }

    return existingNz;
  }
}

unsigned int MultiIndex::NumNz() const
{
    return nzInds.size();
}


unsigned int MultiIndex::MultiIndex::Get(unsigned ind) const
{
  if(ind>length){
    throw std::out_of_range("Tried to access index " + std::to_string(ind) + " of a multiindex with only " + std::to_string(length) + " components.");
    return 0;

  }else{
      auto indIter = std::lower_bound(nzInds.begin(), nzInds.end(), ind);

      if(indIter!=nzInds.end()){
        if((*indIter)==ind){
            unsigned int index = std::distance(nzInds.begin(), indIter);
            return nzVals.at(index);
        }
      }
      return 0;
  }
}



bool MultiIndex::operator!=(const MultiIndex &b) const {

  if( (b.length != length) || (b.maxValue != maxValue) || (b.totalOrder != totalOrder))
    return true;

  if(b.nzInds.size() != nzInds.size())
    return true;

  for(unsigned int i=0; i<nzInds.size(); ++i){
      if( b.nzInds.at(i) != nzInds.at(i))
          return true;
      if( b.nzVals.at(i) != nzVals.at(i))
          return true;
  }

  return false;
}

bool MultiIndex::operator==(const MultiIndex &b) const{
  return !( *this != b);
}

bool MultiIndex::operator>(const MultiIndex &b) const{
  return b<(*this);
}

bool MultiIndex::operator<(const MultiIndex &b) const{

  if(length<b.length){
    return true;
  }else if(length>b.length){
    return false;
  }else if(totalOrder<b.totalOrder){
    return true;
  }else if(totalOrder>b.totalOrder){
    return false;
  }else if(maxValue<b.maxValue){
    return true;
  }else if(maxValue>b.maxValue){
    return false;
  }else{

    for(unsigned int i=0; i<std::min<unsigned int>(length, b.length); ++i){
      if(Get(i)<b.Get(i)){
        return true;
      }else if(Get(i)>b.Get(i)){
        return false;
      }
    }

    // it should never get to this point unless the multiindices are equal
    return false;
  }

}

bool MultiIndex::operator>=(const MultiIndex &b) const{
    return !(*this < b);
}

bool MultiIndex::operator<=(const MultiIndex &b) const{
    return !(*this > b);
}

bool MultiIndex::AnyBounded(const MultiIndex &bound) const{
  if(length > bound.length) {
    throw std::invalid_argument("MultiIndex::AnyExceed: invalid length");
  }
  for(unsigned int i=0; i<length; ++i){
    if(Get(i)>=bound.Get(i)){
      return true;
    }
  }
  return false;
}

bool MultiIndex::HasNonzeroEnd() const{
  return (nzInds.size() > 0) && (nzInds.back() == length - 1);
}

std::string MultiIndex::String() const {
  std::string out;
  for(unsigned int i=0; i<Length(); ++i){
    if (i > 0)
      out += " ";
    out += std::to_string(Get(i));
  }
  return out;
}

std::ostream& mpart::operator<< (std::ostream &out, MultiIndex const& ind)
{
  out << ind.String();
  return out;
}
