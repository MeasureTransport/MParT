#include "MParT/MapFactory.h"

#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/SummarizedMap.h"
#include "MParT/AffineFunction.h"
#include "MParT/IdentityMap.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/RectifiedMultivariateExpansion.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/LinearizedBasis.h"
#include "MParT/Sigmoid.h"
#include "MParT/UnivariateExpansion.h"

using namespace mpart;


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateComponent(FixedMultiIndexSet<MemorySpace> const& mset,
                                                           MapOptions                                   opts)
{
    return CompFactoryImpl<MemorySpace>::GetFactoryFunction(opts)(mset,opts);
}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int dim,
                                                                                         unsigned int activeInd,
                                                                                         std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp)
{

    // Check that active index is not greater than map dimension
    if(dim < activeInd){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the active index can't be greater than map dimension. Got dim = " << dim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    // Check that the input dimension of the component matches the activeInd
    if(activeInd != comp->inputDim){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the component input dimension must be equal to the active index. Got dim = " << comp->inputDim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;
    // Construct map using TriangularMap constructor

    if(activeInd == 1){  // special case if activeInd = 1, map is of form [T_1; Id]

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = comp;
        blocks.at(1) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);


    }
    else if (activeInd == dim){  // special case if activeInd = dim, map is of form [Id; T_d]
        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);
    }
    else{ // general case, map is of form [Id; T_i; Id]

        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(3);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;
        blocks.at(2) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);

    }

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;

}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateTriangular(unsigned int inputDim,
                                                                         unsigned int outputDim,
                                                                         unsigned int totalOrder,
                                                                         MapOptions options)
{

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> comps(outputDim);

    unsigned int extraInputs = inputDim - outputDim;

    for(unsigned int i=0; i<outputDim; ++i){
        FixedMultiIndexSet<Kokkos::HostSpace> mset(i+extraInputs+1, totalOrder);
        comps.at(i) = CreateComponent<MemorySpace>(mset.ToDevice<MemorySpace>(), options);
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(comps);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}


template<typename MemorySpace>
std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> mpart::MapFactory::CreateExpansion(unsigned int outputDim,
                                                                                           FixedMultiIndexSet<MemorySpace> const& mset,
                                                                                           MapOptions                                   opts)
{
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> output;

    if(opts.basisType==BasisTypes::ProbabilistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite> basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<ProbabilistHermite>> basis1d(LinearizedBasis(ProbabilistHermite(opts.basisNorm), opts.basisLB, opts.basisUB));
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::PhysicistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,PhysicistHermite> basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<PhysicistHermite>> basis1d(PhysicistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::HermiteFunctions){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,HermiteFunction> basis1d;
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<HermiteFunction>> basis1d(opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }

    if(output){
        output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
        return output;
    }

    std::stringstream msg;
    msg << "Could not parse options in CreateExpansion.  Unknown 1d basis type.";
    throw std::runtime_error(msg.str());

    return nullptr;
}


template <typename MemorySpace, typename SigmoidType, typename EdgeType>
Sigmoid1d<MemorySpace, SigmoidType, EdgeType> CreateSigmoid(
    unsigned int inputDim, StridedVector<double, MemorySpace> centers, double edgeShape) {
    Kokkos::View<double*, MemorySpace> widths("Sigmoid Widths", centers.size());
    Kokkos::View<double*, MemorySpace> weights("Sigmoid Weights", centers.size());
    int sigmoid_count = centers.size() - 2;
    double max_order_double = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>::LengthToOrder(sigmoid_count);
    int max_order = int(max_order_double);
    if(max_order_double  != double(max_order)) {
        std::stringstream ss;
        ss << "Incorrect length of centers, " << centers.size() << ".\n";
        ss << "Length should be of form 2+(1+2+3+...+n) for some order n";
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(
            ss.str().c_str());
    }
    Kokkos::parallel_for(max_order+2, KOKKOS_LAMBDA(unsigned int i) {
        if (i == max_order) {
            widths(0) = edgeShape;
            weights(0) = 1./edgeShape;
            return;
        } else if (i == max_order+1) {
            widths(1) = edgeShape;
            weights(1) = 1./edgeShape;
            return;
        }
        int start_idx = 2+(i*(i+1))/2;

        for(unsigned int j = 0; j < i; j++) {
            double prev_center, next_center;
            if(j == 0 || i == 0) {// Use center for left edge term
                prev_center = centers(0);
            } else {
                prev_center = centers(start_idx+j-1);
            }
            if(j == i-1 || i == 0) { // Use center for right edge term
                next_center = centers(1);
            } else {
                next_center = centers(start_idx+j);
            }
            widths(start_idx + j) = 2/(next_center - prev_center);
            weights(start_idx + j) = 1.;
        }
    });
    Sigmoid1d<MemorySpace, SigmoidType, EdgeType> sig {centers, widths, weights};
    return sig;
}

template<typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
using SigmoidBasisEval = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagEval, Sigmoid1d<MemorySpace, SigmoidType, EdgeType>>>;

template <typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidExpansionTemplate(
    FixedMultiIndexSet<MemorySpace> mset_offdiag, FixedMultiIndexSet<MemorySpace> mset_diag,
    StridedVector<double, MemorySpace> centers, double edgeWidth)
{
    unsigned int inputDim = mset_diag.Length();
    if(inputDim != 1 && inputDim != mset_offdiag.Length() + 1) {
        std::stringstream ss;
        ss << "Mismatched input dimensions for offdiag and diag multiindex sets\n"
        << "offdiag: " << mset_offdiag.Length() << "\n"
        << "diag: " << mset_diag.Length();
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
    using Sigmoid_T = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>;
    using DiagBasisEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagEval, Sigmoid_T>, Rectifier>;
    using OffdiagBasisEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous, OffdiagEval>;
    auto sigmoid = CreateSigmoid<MemorySpace, SigmoidType, EdgeType>(inputDim, centers, edgeWidth);
    if(inputDim == 1) {
        unsigned int maxOrder = mset_diag.Size() - 1;
        auto output = std::make_shared<UnivariateExpansion<MemorySpace, Sigmoid_T>>(maxOrder, sigmoid);
        output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
        return output;
    }
    DiagBasisEval_T diagBasisEval(inputDim, OffdiagEval(), sigmoid);
    using RMVE = RectifiedMultivariateExpansion<MemorySpace, OffdiagEval, Sigmoid_T, Rectifier>;
    unsigned int maxOrder = diagBasisEval.diag_.GetOrder();
    OffdiagBasisEval_T offdiagBasisEval;
    MultivariateExpansionWorker<DiagBasisEval_T,MemorySpace> worker_diag(mset_diag, diagBasisEval);
    MultivariateExpansionWorker<OffdiagBasisEval_T,MemorySpace> worker_offdiag(mset_offdiag, offdiagBasisEval);
    auto output = std::make_shared<RMVE>(worker_offdiag, worker_diag);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}

template <typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidExpansionTemplate(
    unsigned int inputDim, unsigned int totalOrder, StridedVector<double, MemorySpace> centers, double edgeWidth)
{
    using Sigmoid_T = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>;
    if(totalOrder == 0)
        totalOrder = Sigmoid_T::LengthToOrder(centers.size()-2)+3;
    if(inputDim == 1) {
        FixedMultiIndexSet<MemorySpace> mset {1, totalOrder};
        FixedMultiIndexSet<MemorySpace> mset_offdiag {1,0};
        return CreateSigmoidExpansionTemplate<MemorySpace, OffdiagEval, Rectifier, SigmoidType, EdgeType>(
            mset_offdiag, mset, centers, edgeWidth);
    }
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(inputDim, totalOrder, MultiIndexLimiter::NonzeroDiagTotalOrderLimiter(totalOrder));
    FixedMultiIndexSet<MemorySpace> fmset_diag = mset.Fix(true).ToDevice<MemorySpace>();
    FixedMultiIndexSet<MemorySpace> fmset_offdiag {inputDim-1, totalOrder};
    return CreateSigmoidExpansionTemplate<MemorySpace, OffdiagEval, Rectifier, SigmoidType, EdgeType>(
        fmset_offdiag, fmset_diag, centers, edgeWidth);
}

template<typename MemorySpace>
void HandleSigmoidComponentErrors(MapOptions const& opts) {
    // Check that the opts are valid
    if (opts.basisType != BasisTypes::HermiteFunctions) {
        std::string basisString = MapOptions::btypes[static_cast<unsigned int>(opts.basisType)];
        std::stringstream ss;
        ss << "Unsupported basis type for sigmoid expansion: " << basisString;
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
    if(opts.posFuncType != PosFuncTypes::Exp && opts.posFuncType != PosFuncTypes::SoftPlus) {
        std::string posString = MapOptions::pftypes[static_cast<unsigned int>(opts.posFuncType)];
        std::stringstream ss;
        ss << "Unsupported positive function type for sigmoid expansion: " << posString;
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
    if(opts.edgeType != EdgeTypes::SoftPlus) {
        std::string edgeString = MapOptions::etypes[static_cast<unsigned int>(opts.edgeType)];
        std::stringstream ss;
        ss << "Unsupported edge type for sigmoid expansion: " << edgeString;
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
    if(opts.sigmoidType != SigmoidTypes::Logistic) {
        std::string sigmoidString = MapOptions::stypes[static_cast<unsigned int>(opts.sigmoidType)];
        std::stringstream ss;
        ss << "Unsupported sigmoid type for sigmoid expansion: " << sigmoidString;
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidComponent(
    unsigned int inputDim, unsigned int totalOrder, StridedVector<const double, MemorySpace> centers, MapOptions opts) {
    HandleSigmoidComponentErrors<MemorySpace>(opts);
    Kokkos::View<double*, MemorySpace> centers_copy("Centers Copy", centers.size());
    Kokkos::deep_copy(centers_copy, centers);
    // Dispatch to the correct sigmoid expansion template
    if(opts.posFuncType == PosFuncTypes::Exp) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, Exp, SigmoidTypeSpace::Logistic, SoftPlus>(inputDim, totalOrder, centers_copy, opts.edgeShape);
    } else if(opts.posFuncType == PosFuncTypes::SoftPlus) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, SoftPlus, SigmoidTypeSpace::Logistic, SoftPlus>(inputDim, totalOrder, centers_copy, opts.edgeShape);
    }
    else {
        return nullptr;
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidComponent(
    FixedMultiIndexSet<MemorySpace> mset_offdiag, FixedMultiIndexSet<MemorySpace> mset_diag,
    StridedVector<const double, MemorySpace> centers, MapOptions opts) {
    HandleSigmoidComponentErrors<MemorySpace>(opts);
    Kokkos::View<double*, MemorySpace> centers_copy("Centers Copy", centers.size());
    Kokkos::deep_copy(centers_copy, centers);
    // Dispatch to the correct sigmoid expansion template
    if(opts.posFuncType == PosFuncTypes::Exp) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, Exp, SigmoidTypeSpace::Logistic, SoftPlus>(mset_offdiag, mset_diag, centers_copy, opts.edgeShape);
    } else if(opts.posFuncType == PosFuncTypes::SoftPlus) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, SoftPlus, SigmoidTypeSpace::Logistic, SoftPlus>(mset_offdiag, mset_diag, centers_copy, opts.edgeShape);
    }
    else {
        return nullptr;
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidTriangular(unsigned int inputDim,
    unsigned int outputDim, unsigned int totalOrder, std::vector<StridedVector<const double, MemorySpace>> const& centers, MapOptions opts) {
    if(outputDim > inputDim) {
        std::stringstream ss;
        ss << "CreateSigmoidTriangular: Output dimension " << outputDim << " cannot be greater than input dimension " << inputDim;
        ProcAgnosticError<MemorySpace, std::invalid_argument>::error(ss.str().c_str());
    }
    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>> > comps(outputDim);
    for(int i = 0; i < outputDim; i++) {
        StridedVector<const double, MemorySpace> center_view = centers[i];
        unsigned int inputDim_i = (inputDim - outputDim) + i+1;
        auto comp = CreateSigmoidComponent<MemorySpace>(inputDim_i, totalOrder, center_view, opts);
        comps[i] = comp;
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(comps);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}

template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> mpart::MapFactory::CreateExpansion<Kokkos::HostSpace>(unsigned int, FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> const&);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(unsigned int, unsigned int, StridedVector<const double, Kokkos::HostSpace>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace>, FixedMultiIndexSet<Kokkos::HostSpace>, StridedVector<const double, Kokkos::HostSpace>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, std::vector<StridedVector<const double, Kokkos::HostSpace>> const&, MapOptions);
#if defined(MPART_ENABLE_GPU)
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ParameterizedFunctionBase<DeviceSpace>> mpart::MapFactory::CreateExpansion<DeviceSpace>(unsigned int, FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateTriangular<DeviceSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<DeviceSpace>> const&);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(unsigned int, unsigned int, StridedVector<const double, DeviceSpace>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace>, FixedMultiIndexSet<DeviceSpace>, StridedVector<const double, DeviceSpace>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidTriangular<DeviceSpace>(unsigned int, unsigned int, unsigned int, std::vector<StridedVector<const double, DeviceSpace>> const&, MapOptions);
#endif