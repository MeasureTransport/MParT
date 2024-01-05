#ifndef MPART_MAPOBJECTIVE_H
#define MPART_MAPOBJECTIVE_H

#include "ConditionalMapBase.h"
#include "Distributions/PullbackDensity.h"
#include "Utilities/ArrayConversions.h"
#include "Utilities/LinearAlgebra.h"
#include "Distributions/GaussianSamplerDensity.h"

namespace mpart {

/**
 * @brief An abstract class to represent an objective for optimizing a transport map based on a Training (and perhaps testing) dataset.
 * @details MapObjective is a class that represents a functional \f$T\f$ of a transport map \f$T(\cdot;\theta)\f$, expected to be estimated using some dataset \f$\mathcal{S}\f$.
 * It provides facilities to use a training dataset as well as an optional testing dataset, and provides the functionality \f$F(T(\cdot;\theta);\mathcal{S})\f$ and
 * \f$\nabla_\theta F(T(\cdot;\theta);\mathcal{S})\f$, the gradient of the objective with respect to the map coefficients/parameters.
 *
 * @tparam MemorySpace Space where all data is stored
 */
template<typename MemorySpace>
class MapObjective {
    private:
    /**
     * @brief Training dataset for objective
     *
     */
    StridedMatrix<const double, MemorySpace> train_;

    /**
     * @brief Testing dataset for objective
     *
     */
    StridedMatrix<const double, MemorySpace> test_;

    public:
    MapObjective() = default;
    
    /**
     * @brief Construct a new Map Objective object from just a training dataset
     *
     * @param train dataset for calculating the objective value to train off of
     */
    MapObjective(StridedMatrix<const double, MemorySpace> train): train_(train) {}

    /**
     * @brief Construct a new Map Objective object from a training and testing dataset
     *
     * @param train dataset for calculating the objective value to train off of
     * @param test dataset for calculating the objective value when testing how good the map is
     */
    MapObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test): train_(train), test_(test) {}

    virtual ~MapObjective() = default;

    /**
     * @brief Exposed functor-like function to calculate objective value and its gradient
     *
     * @param n Length of input vector
     * @param x Input vector
     * @param grad Where to store the gradient of the objective (length `n`)
     * @param map Map which we optimize on
     * @return double Training error
     */
    double operator()(unsigned int n, const double* x, double* grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map);

    unsigned int InputDim() const {return train_.extent(0);}
    virtual unsigned int MapOutputDim() const {return train_.extent(0);}
    unsigned int NumSamples() const {return train_.extent(1);}

    /**
     * @brief Shortcut to calculate the error of the map on the training dataset
     *
     * @param map Map to calculate the error on
     * @return double training error
     */
    double TrainError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;

    /**
     * @brief Shortcut to calculate the error of the map on the testing dataset
     *
     * @param map Map to calculate the error on
     * @return double testing error
     */
    double TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;

    /**
     * @brief Get the gradient of the map objective with respect to map coefficients on the training dataset
     *
     * @param map Map to use in this objective gradient
     * @return StridedVector<double, MemorySpace> Gradient of the map on the training dataset
     */
    StridedVector<double, MemorySpace> TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const;

    /**
     * @brief Shortcut to calculate the gradient of the objective on the training dataset w.r.t. the map coefficients
     *
     * @param map Map to calculate the gradient with respect to
     * @param grad storage for the gradient
     */
    void TrainCoeffGradImpl(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, StridedVector<double, MemorySpace> grad) const;

    /**
     * @brief Get the Training data for this objective
     *
     * @return StridedMatrix<const double, MemorySpace> Training data for optimization
     */
    StridedMatrix<const double, MemorySpace> GetTrain() {return train_;}

    /**
     * @brief Get the Testing data for this objective
     *
     * @return StridedMatrix<const double, MemorySpace> Testing data for optimization
     */
    StridedMatrix<const double, MemorySpace> GetTest() {return test_;}

    /**
     * @brief Objective value of map at data
     *
     * @param data dataset to take the objective value on
     * @param map map to calculate objective with respect to
     * @return double Objective value of map at data
     */
    virtual double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const = 0;

    /**
     * @brief Gradient of the objective at the data with respect to the coefficients of the map
     *
     * @param data dataset to calculate the gradient over
     * @param grad storage for calculating gradient inplace
     * @param map map with coefficients to take gradient on
     */
    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const = 0;

    /**
     * @brief Implementation of objective and gradient objective calculation (gradient w.r.t. map coefficients), inplace. Default uses `ObjectiveImpl` and `CoeffGradImpl`,
     *          but best performance should be custom-implemented.
     *
     * @param data Dataset to take the objective and gradient w.r.t the map over
     * @param grad Storage for calculating the gradient of the map w.r.t map coefficients
     * @param map Map to evaluate on
     * @return double objective value on the map at the dataset
     */
    virtual double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const{
        CoeffGradImpl(data, grad, map);
        return ObjectiveImpl(data, map);
    }
};

/**
 * @brief Calculate the sample-based forward Kullback-Leibler divergence of a map w.r.t to a given density.
 * @details For a given function \f$T:\mathbb{R}^n\to\mathbb{R}^m\f$ with \f$m\leq n\f$, and a dataset \f$\mathcal{S} = \{X^{(k)}\}\subset\mathbb{R}^n\sim\nu\f$,
 * estimate the sample-based forward KL divergence \f$\hat{D}(T;\mathcal{S})\approx D(T^\sharp\nu||\mu)\f$ using the empirical distribution of \f$\{X^{(k)}\}\f$,
 * where \f$\mu\f$ is a measure with some density \f$\pi\f$. Explicitly, \f$\hat{D}(T;\mathcal{S}) := -\frac{1}{K}\sum_{k=1}^K \log p(T(X^{(k)})) + \log|\det(\nabla T(X^{(k)}))\f$.
 *
 * @tparam MemorySpace Space where data is stored
 * @see mpart::MapObjective
 */
template<typename MemorySpace>
class KLObjective: public MapObjective<MemorySpace> {
    public:
    /**
     * @brief Construct a new KLObjective object from a training dataset and a density for KL
     *
     * @param train Dataset for training the map
     * @param density Density \f$\mu\f$ to calculate the KL with respect to (i.e. \f$D(\cdot||\mu)\f$ )
     */
    KLObjective(StridedMatrix<const double, MemorySpace> train, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective<MemorySpace>(train), density_(density) {}
    /**
     * @brief Construct a new KLObjective object from training and testing datasets and a density for KL
     *
     * @param train Dataset for training the map
     * @param test Dataset for testing the map
     * @param density Density \f$\mu\f$ to calculate the KL with respect to (i.e. \f$D(\cdot||\mu)\f$ )
     */
    KLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, std::shared_ptr<DensityBase<MemorySpace>> density): MapObjective<MemorySpace>(train, test), density_(density) {}

    double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    unsigned int MapOutputDim() const override {return density_->Dim();}
    private:
    /**
     * @brief Density \f$\mu\f$ to calculate the KL with respect to (i.e. \f$D(\cdot||\mu)\f$ )
     *
     */
    std::shared_ptr<DensityBase<MemorySpace>> density_;
};

template<typename MemorySpace>
class ParamL2RegularizationObjective: public MapObjective<MemorySpace> {
    public:
    ParamL2RegularizationObjective(double scale): MapObjective<MemorySpace>(), scale_(scale) {}
    double ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    double ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;
    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const override;

    /// @brief Change the scale parameter of the regularization
    /// @param new_scale
    void SetScale(double new_scale) {scale_ = new_scale;}

    private:
    double scale_;
};

namespace ObjectiveFactory {
template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, unsigned int dim=0);

template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, unsigned int dim=0);
} // namespace ObjectiveFactory

} // namespace mpart

#endif //MPART_MAPOBJECTIVE_H