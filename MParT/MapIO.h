#ifndef MPART_MapIO_H
#define MPART_MapIO_H

#include <iostream>
#include "MParT/MapOptions.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
using namespace mpart {

int SaveMapInfo(std::string const& filename, MapOptions const& options,
                FixedMultiIndexSet const& mset, StridedVector<double> const& coeffs);

int LoadMapInfo(std::string const& filename, MapOptions& options,
                FixedMultiIndexSet& mset, StridedVector<double>& coeffs);

} // namespace mpart

#endif // MPART_MapIO_H