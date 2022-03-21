#include "MParT/MultiIndices/MultiIndexLimiter.h"

#include <cmath>
#include <stdexcept>

using namespace mpart;
using namespace mpart::MultiIndexLimiter;


bool Dimension::operator()(MultiIndex const& multi) const
{
    for(unsigned int i=0; i<lowerDim; ++i)
    {
        if(multi.Get(i)!=0)
            return false;
    }
    for(unsigned int i=lowerDim+length; i<multi.Length(); ++i){
        if(multi.Get(i)!=0)
            return false;
    }
    return true;
}



Anisotropic::Anisotropic(std::vector<double> const& weightsIn,
                         double                     epsilonIn) : weights(weightsIn),
                                                                 epsilon(epsilonIn)
{
    // validate weight vector
  for(int i = 0; i < weights.size(); ++i){
    if((weights.at(i) > 1.0) || (weights.at(i) < 0.0))
      throw std::invalid_argument("AnisotropicLimiter requires all weights have to be in [0,1]. Got weight " + std::to_string(weights[i]));
  }

  // validate threshold
  if ((epsilon >= 1.0) || (epsilon <= 0.0))
      throw std::invalid_argument("AnisotropicLimiter requires epsilon to be in (0,1). Got epsilon = " + std::to_string(epsilon));
}


bool Anisotropic::operator()(MultiIndex const& multi) const
{

    if(multi.Length() != weights.size())
        return false;

    double prod = 1;
    for(unsigned int i=0; i<multi.Length(); ++i)
        prod *= pow(weights.at(i), (int)multi.Get(i));

    return prod >= epsilon;
}


bool MaxDegree::operator()(MultiIndex const& multi) const
{
    if(multi.Length() != maxDegrees.size())
        return false;

    for(unsigned int i=0; i<multi.Length(); ++i){
        if(multi.Get(i)>maxDegrees.at(i))
            return false;
    }
    return true;
}


