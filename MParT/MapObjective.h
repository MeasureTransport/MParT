#ifndef MPART_MAPOBJECTIVE_H
#define MPART_MAPOBJECTIVE_H

#include "ConditionalMapBase.h"
#include "Distributions/PullbackDensity.h"
#include "Utilities/ArrayConversions.h"
#include "Utilities/LinearAlgebra.h"

namespace mpart {
template<typename MemorySpace>
class MapObjective {
    private:
    StridedMatrix<const double, MemorySpace> train_;
    StridedMatrix<const double, MemorySpace> test_;

    public:
    MapObjective() = delete;
    MapObjective(StridedMatrix<const double, MemorySpace> train): train_(train) {}
    MapObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test): train_(train), test_(test) {}

    double operator()(unsigned int n, const double* x, double* grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map);

    unsigned int Dim(){return train_.extent(0);}
    unsigned int NumSamples(){return train_.extent(1);}
    double TrainError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;
    double TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;

    StridedVector<double, MemorySpace> TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;
    void TrainCoeffGradImpl(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, StridedVector<double, MemorySpace> grad) const;

    StridedMatrix<const double, MemorySpace> GetTrain() {return train_;}
    StridedMatrix<const double, MemorySpace> GetTest() {return test_;}

    virtual double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const = 0;
    virtual double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const = 0;
    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const = 0;
};

template<typename MemorySpace>
class KLObjective: public MapObjective<MemorySpace> {
    public:
    KLObjective(StridedMatrix<const double, MemorySpace> train, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective<MemorySpace>(train), density_(density) {}
    KLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective<MemorySpace>(train, test), density_(density) {}

    double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;

    private:
    std::shared_ptr<DensityBase<MemorySpace>> density_;
};

} // namespace mpart

#endif //MPART_MAPOBJECTIVE_H