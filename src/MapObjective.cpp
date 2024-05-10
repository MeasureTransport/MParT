#include "MParT/MapObjective.h"

using namespace mpart;

template<typename MemorySpace>
double MapObjective<MemorySpace>::operator()(unsigned int n, const double* coeffs, double* grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {

    Kokkos::View<const double*, MemorySpace> coeffView = ToConstKokkos<double,MemorySpace>(coeffs, n);
    if (grad == nullptr) return ObjectiveImpl(train_, map);
    StridedVector<double, MemorySpace> gradView = ToKokkos<double,MemorySpace>(grad, n);
    map->SetCoeffs(coeffView);
    return ObjectivePlusCoeffGradImpl(train_, gradView, map);
}

template<typename MemorySpace>
double MapObjective<MemorySpace>::TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    if(test_.extent(0) == 0) {
        throw std::runtime_error("No test dataset given!");
    }
    return ObjectiveImpl(test_, map);
}

template<typename MemorySpace>
double MapObjective<MemorySpace>::TrainError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    return ObjectiveImpl(train_, map);
}

template<typename MemorySpace>
void MapObjective<MemorySpace>::TrainCoeffGradImpl(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, StridedVector<double, MemorySpace> grad) const {
    CoeffGradImpl(train_, grad, map);
}

template<typename MemorySpace>
StridedVector<double, MemorySpace> MapObjective<MemorySpace>::TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    Kokkos::View<double*, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
    TrainCoeffGradImpl(map, grad);
    return grad;
}

template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> ObjectiveFactory::CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, unsigned int dim) {
    if(dim == 0) dim = train.extent(0);
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
    return std::make_shared<KLObjective<MemorySpace>>(train, density);
}

template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> ObjectiveFactory::CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, unsigned int dim) {
    if(dim == 0) dim = train.extent(0);
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
    return std::make_shared<KLObjective<MemorySpace>>(train, test, density);
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    using ExecSpace = typename MemoryToExecution<MemorySpace>::Space;
    unsigned int N_samps = data.extent(1);
    unsigned int grad_dim = grad.extent(0);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    double sumDensity = 0.;
   
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,N_samps); 
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", policy, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);


    if (grad.data()!=nullptr){
        double scale = -1.0/((double) N_samps);

        Kokkos::TeamPolicy<ExecSpace> policy(grad_dim, Kokkos::AUTO());
        using team_handle = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
        Kokkos::parallel_for(policy,
            KOKKOS_LAMBDA(const team_handle& teamMember){
                int row = teamMember.league_rank();
                double thisRowSum = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N_samps), 
                [&] (const int& col, double& innerUpdate) {
                    innerUpdate += scale*densityGradX(row,col);
                },
                thisRowSum);

                grad(row) = thisRowSum;
            }
        );
        Kokkos::fence();
    }
    return sumDensity/N_samps;
}

template<typename MemorySpace>
FastGaussianReverseKLObjective<MemorySpace>::FastGaussianReverseKLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, unsigned int numCoeffs) : MapObjective<MemorySpace>(train, test) {
    unsigned int N = std::max(train.extent(1), test.extent(1));
    eval_space_ = Kokkos::View<double**, MemorySpace>("eval_space", 1, N);
    logdet_space_ = Kokkos::View<double*, MemorySpace>("logdet_space", N);
    grad_space_ = Kokkos::View<double**, MemorySpace>("grad_space", numCoeffs, N);
    logdet_grad_space_ = Kokkos::View<double**, MemorySpace>("logdet_grad_space", numCoeffs, N);
}

template<typename MemorySpace>
FastGaussianReverseKLObjective<MemorySpace>::FastGaussianReverseKLObjective(StridedMatrix<const double, MemorySpace> train, unsigned int numCoeffs) : FastGaussianReverseKLObjective<MemorySpace>(train, Kokkos::View<const double**, MemorySpace>(), numCoeffs) {}

template<typename MemorySpace>
template<unsigned int Type_idx>
void FastGaussianReverseKLObjective<MemorySpace>::FillSpaces(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, StridedMatrix<const double, MemorySpace> data) const {
    constexpr ObjectiveType Type = static_cast<ObjectiveType>(Type_idx);
    unsigned int N_points = data.extent(1);
    if(N_points > eval_space_.extent(1)){
        std::stringstream ss;
        ss << "FastGaussianReverseKLObjective: Not enough space allocated for evaluation!" <<
            "Need " << N_points << " points, storing " << eval_space_.extent(1) << " points.";
        throw std::invalid_argument(ss.str().c_str());
    }
    StridedMatrix<double, MemorySpace> eval_space_view = Kokkos::subview(eval_space_, Kokkos::ALL(), Kokkos::make_pair(0u, N_points));
    map->EvaluateImpl(data, eval_space_view);
    if constexpr((Type == ObjectiveType::Eval) || (Type == ObjectiveType::EvalGrad)) {
        StridedVector<double, MemorySpace> logdet_space_view = Kokkos::subview(logdet_space_, Kokkos::make_pair(0u, N_points));
        map->LogDeterminantImpl(data, logdet_space_view);
    }
    if constexpr((Type == ObjectiveType::Grad) || (Type == ObjectiveType::EvalGrad)) {
        StridedMatrix<double, MemorySpace> grad_space_view = Kokkos::subview(grad_space_, Kokkos::ALL(), Kokkos::make_pair(0u, N_points));
        StridedMatrix<double, MemorySpace> logdet_grad_space_view = Kokkos::subview(logdet_grad_space_, Kokkos::ALL(), Kokkos::make_pair(0u, N_points));
        Kokkos::fence();
        map->CoeffGradImpl(data, eval_space_view, grad_space_view);
        map->LogDeterminantCoeffGradImpl(data, logdet_grad_space_view);
    }
    Kokkos::fence();
}

template<typename MemorySpace>
template<unsigned int Type_idx>
void FastGaussianReverseKLObjective<MemorySpace>::ClearSpaces() const {
    constexpr ObjectiveType Type = static_cast<ObjectiveType>(Type_idx);
    if constexpr((Type == ObjectiveType::Eval) || (Type == ObjectiveType::EvalGrad)) {
        Kokkos::deep_copy(eval_space_, 0.);
        Kokkos::deep_copy(logdet_space_, 0.);
    }
    if constexpr((Type == ObjectiveType::Grad) || (Type == ObjectiveType::EvalGrad)) {
        Kokkos::deep_copy(grad_space_, 0.);
        Kokkos::deep_copy(logdet_grad_space_, 0.);
    }
}

template<typename EvalSpaceType, typename LogDetSpaceType>
class EvaluationFunctor {
    public:
    EvaluationFunctor(const EvalSpaceType& eval_space_view, const LogDetSpaceType& logdet_space_view): eval_space_view_(eval_space_view), logdet_space_view_(logdet_space_view){}
    KOKKOS_FUNCTION void operator()(const unsigned int j, double& loss) const {
        loss += 0.5*eval_space_view_(0, j)*eval_space_view_(0, j) - logdet_space_view_(j);
    }
    private:
    const EvalSpaceType eval_space_view_;
    const LogDetSpaceType logdet_space_view_;
};

template<typename KLGradSpaceType, typename GradSpaceType, typename GradLogDetSpaceType>
class GradientFunctor {
    using team_handle=typename Kokkos::TeamPolicy<typename GradSpaceType::execution_space>::member_type;
    public:
    GradientFunctor(KLGradSpaceType& kl_grad, unsigned int N_points, const GradSpaceType& grad_space_view, const GradLogDetSpaceType& logdet_grad_space_view):
        kl_grad_(kl_grad), N_points_(N_points), grad_space_view_(grad_space_view), logdet_grad_space_view_(logdet_grad_space_view) {
    }
    KOKKOS_FUNCTION void operator()(const team_handle& team) const {
        int d = team.league_rank();
        double grad_d_ = 0.;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, N_points_), [&](const unsigned int j, double& grad_d_tmp) {
            grad_d_tmp += grad_space_view_(d, j) - logdet_grad_space_view_(d, j);
        }, grad_d_);
        kl_grad_(d) = grad_d_/N_points_;
    }
    
    private:
    mutable KLGradSpaceType kl_grad_;
    const unsigned int N_points_;
    const GradSpaceType grad_space_view_;
    const GradLogDetSpaceType logdet_grad_space_view_;
};

template<typename MemorySpace>
template<unsigned int Type_idx>
double FastGaussianReverseKLObjective<MemorySpace>::CommonEval(StridedMatrix<const double, MemorySpace> points, StridedVector<double, MemorySpace> kl_grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    if(map->outputDim != 1){
        throw std::invalid_argument("FastGaussianReverseKLObjective: Map output dimension must be 1! Found dimension " + std::to_string(map->outputDim) + ".");
    }
    constexpr ObjectiveType Type = static_cast<ObjectiveType>(Type_idx);
    unsigned int N_points = points.extent(1);
    unsigned int numCoeffs = map->numCoeffs;
    if(Type != ObjectiveType::Eval && numCoeffs != kl_grad.extent(0)){
        std::stringstream ss;
        ss << "FastGaussianReverseKLObjective: Gradient vector has incorrect size!"
           << "Given size " << kl_grad.extent(0) << ", expected size " << numCoeffs
           << ".";
        throw std::invalid_argument(ss.str().c_str());
    }
    FillSpaces<Type_idx>(map, points);

    double kl_loss = 0.;
    if constexpr((Type == ObjectiveType::Eval) || (Type == ObjectiveType::EvalGrad)) {
        Kokkos::RangePolicy<ExecSpace> eval_policy(0, N_points);
        EvaluationFunctor eval_kernel(eval_space_, logdet_space_);
        Kokkos::parallel_reduce("FastGaussianReverseKL Evaluate", eval_policy, eval_kernel, kl_loss);
        kl_loss /= N_points;
    }
    if constexpr((Type == ObjectiveType::Grad) || (Type == ObjectiveType::EvalGrad)) {
        Kokkos::TeamPolicy<ExecSpace> grad_policy(numCoeffs, Kokkos::AUTO());
        GradientFunctor gradient_functor(kl_grad, N_points, grad_space_, logdet_grad_space_);

        Kokkos::parallel_for("FastGaussianReverseKL Gradient", grad_policy, gradient_functor);
        Kokkos::fence();
    }
    ClearSpaces<Type_idx>();
    return kl_loss;
}

template<typename MemorySpace>
double FastGaussianReverseKLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    constexpr unsigned int EvalGrad_idx = static_cast<unsigned int>(ObjectiveType::EvalGrad);
    return CommonEval<EvalGrad_idx>(data, grad, map);
}

template<typename MemorySpace>
double FastGaussianReverseKLObjective<MemorySpace>::ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    Kokkos::View<double*, MemorySpace> grad_holder;
    constexpr unsigned int Eval_idx = static_cast<unsigned int>(ObjectiveType::Eval);
    return CommonEval<Eval_idx>(data, grad_holder, map);
}

template<typename MemorySpace>
void FastGaussianReverseKLObjective<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    constexpr unsigned int Grad_idx = static_cast<unsigned int>(ObjectiveType::Grad);
    CommonEval<Grad_idx>(data, grad, map);
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    double sumDensity = 0.;
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,N_samps); 
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", policy, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    return sumDensity/N_samps;
}

template<typename MemorySpace>
void KLObjective<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    using ExecSpace = typename MemoryToExecution<MemorySpace>::Space;
    unsigned int N_samps = data.extent(1);
    unsigned int grad_dim = grad.extent(0);
    PullbackDensity<MemorySpace> pullback {map, density_}; 
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);

    double scale = -1.0/((double) N_samps);
    Kokkos::TeamPolicy<ExecSpace> policy(grad_dim, Kokkos::AUTO());
    using team_handle = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(policy,
        KOKKOS_LAMBDA(const team_handle& teamMember){
            int row = teamMember.league_rank();
            double thisRowSum = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N_samps), 
            [&] (const int& col, double& innerUpdate) {
                innerUpdate += scale*densityGradX(row,col);
            },
            thisRowSum);
            grad(row) = thisRowSum;
        }
    );
    Kokkos::fence();
}

// Explicit template instantiation
template class mpart::MapObjective<Kokkos::HostSpace>;
template class mpart::KLObjective<Kokkos::HostSpace>;
template class mpart::FastGaussianReverseKLObjective<Kokkos::HostSpace>;
template std::shared_ptr<MapObjective<Kokkos::HostSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
template std::shared_ptr<MapObjective<Kokkos::HostSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace>, StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
#if defined(MPART_ENABLE_GPU)
    template class mpart::MapObjective<DeviceSpace>;
    template class mpart::KLObjective<DeviceSpace>;
    template class mpart::FastGaussianReverseKLObjective<DeviceSpace>;
    template std::shared_ptr<MapObjective<DeviceSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(StridedMatrix<const double, DeviceSpace>, unsigned int);
    template std::shared_ptr<MapObjective<DeviceSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(StridedMatrix<const double, DeviceSpace>, StridedMatrix<const double, DeviceSpace>, unsigned int);
#endif
