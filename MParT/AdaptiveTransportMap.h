#ifndef MPART_ADAPTIVETRANSPORTMAP_H
#define MPART_ADAPTIVETRANSPORTMAP_H

#include <vector>
#include <nlopt/nlopt.hpp>
#include "MapOptions.h"

double negativeLogLikelihood(const std::vector<double> &x, std::vector<double> &grad, void *data);

enum class ReferenceTypes {
    StandardGaussian
};

struct ATMOptions: public MapOptions {
    int maxPatience = 10;
    int max_terms = 10;
    ReferenceTypes = ReferenceTypes::StandardGaussian;

};

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(MultiIndexSet &mset0, StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions());

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> AdaptiveTransportMap(StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x, ATMOptions options = ATMOptions());


#endif // MPART_ADAPTIVETRANSPORTMAP_H