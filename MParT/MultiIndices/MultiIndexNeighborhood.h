#ifndef MPART_MULTIINDEXNEIGHBORHOOD_H
#define MPART_MULTIINDEXNEIGHBORHOOD_H

#include <vector>

#include "MParT/MultiIndices/MultiIndex.h"

namespace mpart{

/**
  @brief Abstract base class for MultiIndexSet graph connectivity definitions.
 */
class MultiIndexNeighborhood{
public:
    virtual ~MultiIndexNeighborhood() = default;

    virtual std::vector<MultiIndex> ForwardNeighbors(MultiIndex const& multi) = 0;

    virtual std::vector<MultiIndex> BackwardNeighbors(MultiIndex const& multi) = 0;

    virtual bool IsForward(MultiIndex const& base, MultiIndex const& next) = 0;
    virtual bool IsBackward(MultiIndex const& base, MultiIndex const& next) = 0;


};

/**
 @brief Defines the standard graph connectivity for a MultiIndexSet.
  A multiindex \f$\mathbf{\alpha}\f$ is a **forward** neighbor of \f$\mathbf{\beta}\f$ if
  \f$\|\mathbf{\alpha}-\mathbf{\beta}|_1==1\f$ and \f$\mathbf{\alpha}_i \geq \mathbf{\alpha}_i\f$ for all \f$i\f$.

  A multiindex \f$\mathbf{\alpha}\f$ is a **backward** neighbor of \f$\mathbf{\beta}\f$ if
  \f$\|\mathbf{\alpha}-\mathbf{\beta}|_1==1\f$ and \f$\mathbf{\alpha}_i \leq \mathbf{\alpha}_i\f$ for all \f$i\f$.
 */
class DefaultNeighborhood : public MultiIndexNeighborhood{
public:

    virtual ~DefaultNeighborhood() = default;

    virtual std::vector<MultiIndex> ForwardNeighbors(MultiIndex const& multi) override;

    virtual std::vector<MultiIndex> BackwardNeighbors(MultiIndex const& multi) override;

    virtual bool IsForward(MultiIndex const& base,
                           MultiIndex const& next) override;

    virtual bool IsBackward(MultiIndex const& base,
                            MultiIndex const& prev) override;
};

}


#endif