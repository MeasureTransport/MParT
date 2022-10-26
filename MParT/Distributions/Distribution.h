#ifndef MPART_Distribution_H
#define MPART_Distribution_H

#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/DensityBase.h"

namespace mpart {

template<typename MemorySpace>
class Distribution: public SampleGenerator<MemorySpace>, DensityBase<MemorySpace> {};

} // namespace mpart

#endif //MPART_Distribution_H