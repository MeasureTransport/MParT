#ifndef MPART_MAPOBJECTIVE_H
#define MPART_MAPOBJECTIVE_H

#include "Utilities/ArrayConversions.h"

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

    virtual std::function<double(unsigned int, const double*, double*)> GetOptimizationObjective(std::shared_ptr<ConditionalMapBase<MemorySpace>> map);

    double TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map);

    StridedVector<double, MemorySpace> TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map);

    protected:
    virtual double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) = 0;
    virtual double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) = 0;
    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) = 0;
};

template<typename MemorySpace>
class KLObjective: public MapObjective<MemorySpace> {
    public:
    KLObjective(StridedMatrix<const double, MemorySpace> train, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective(train), density_(density) {}
    KLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective(train, test), density_(density) {}

    protected:
    double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) override;
    double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) override;
    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) override;

    private:
    std::shared_ptr<DensityBase<MemorySpace>> density_;
}

} // namespace mpart

#endif //MPART_MAPOBJECTIVE_H