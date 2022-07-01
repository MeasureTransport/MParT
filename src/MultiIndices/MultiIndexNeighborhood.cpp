#include "MParT/MultiIndices/MultiIndexNeighborhood.h"

using namespace mpart;


std::vector<MultiIndex> DefaultNeighborhood::ForwardNeighbors(MultiIndex const& multi)
{
    std::vector<MultiIndex> output;
    std::vector<unsigned int> vec = multi.Vector();

    for(unsigned int i=0;i<vec.size(); ++i){
        vec.at(i)++;
        output.push_back(MultiIndex(vec));
        vec.at(i)--;
    }

    return output;
}

std::vector<MultiIndex> DefaultNeighborhood::BackwardNeighbors(MultiIndex const& multi)
{
    std::vector<MultiIndex> output;
    std::vector<unsigned int> vec = multi.Vector();

    for(unsigned int i=0; i<vec.size(); ++i){
        if(vec.at(i)!=0){
            vec.at(i)--;
            output.push_back(MultiIndex(vec));
            vec.at(i)++;
        }
    }

    return output;
}

bool DefaultNeighborhood::IsForward(MultiIndex const& base,
                                    MultiIndex const& next)
{

    // Do a first check just based on the number of nonzeros
    const unsigned int nextNnz = next.NumNz();
    const unsigned int baseNnz = base.NumNz();
    if((nextNnz>baseNnz+1) || (nextNnz<baseNnz))
        return false;


    // Now check each dimension
    const unsigned int length = next.Length();
    unsigned int nextVal, baseVal;
    unsigned int diffSum = 0;
    for(unsigned int i=0; i<length; ++i){
        nextVal = next.Get(i);
        baseVal = base.Get(i);
        if(nextVal<baseVal)
            return false;

        diffSum += (nextVal-baseVal);

        if(diffSum>1)
            return false;
    }

    return (diffSum==1);
}

bool DefaultNeighborhood::IsBackward(MultiIndex const& base,
                                     MultiIndex const& prev)
{
    return IsForward(prev,base);
}
