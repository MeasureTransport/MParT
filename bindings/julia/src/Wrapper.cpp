#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/MapOptions.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/MapFactory.h"

#include "CommonJuliaUtilities.h"
#include "CommonUtilities.h"

#include <MParT/MapOptions.h>
#include <Kokkos_Core.hpp>
#include <tuple>
#include <iostream>
#include <memory>

#include "JlArrayConversions.h"
#include "CommonJuliaUtilities.h"

namespace jlcxx {
    template<> struct IsMirroredType<mpart::MultiIndexLimiter::None> : std::false_type { };
    template<> struct SuperType<mpart::ConditionalMapBase<Kokkos::HostSpace>> {typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
    template<> struct SuperType<mpart::TriangularMap<Kokkos::HostSpace>> {typedef mpart::ConditionalMapBase<Kokkos::HostSpace> type;};
    
    
}

namespace mpart {
    jlcxx::ArrayRef<double> EvaluateJulia(ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double,2> pts) {
        unsigned int numPts = size(pts,1);
        unsigned int outDim = map.outputDim;
        jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outDim, numPts);
        map.EvaluateImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
        return output;
    }

    jlcxx::Array<double> CoeffMapJulia(ParameterizedFunctionBase<Kokkos::HostSpace> &pfb){ return KokkosToJulia(pfb.Coeffs()); }
    void SetCoeffsJulia(ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double> v){ pfb.SetCoeffs(JuliaToKokkos(v)); }
    unsigned int numCoeffsJulia(ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.numCoeffs; }
    unsigned int inputDimJulia(ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.inputDim; }
    unsigned int outputDimJulia(ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.outputDim; }
}

using namespace mpart;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    // CommonUtilitiesWrapper(mod);
    // MultiIndexWrapper(mod);
    // MapOptionsWrapper(mod);

    mod.method("Initialize", [](){mpart::binding::Initialize(std::vector<std::string> {});});


    mod.add_type<MultiIndex>("MultiIndex")
        .constructor()
        .constructor<unsigned int, unsigned int>()
        .constructor<unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("NumNz", &MultiIndex::NumNz)
        .method("count_nonzero", &MultiIndex::NumNz);

    jlcxx::stl::apply_stl<MultiIndex>(mod);

    mod.add_type<Kokkos::HostSpace>("HostSpace");
    mod.add_type<Kokkos::LayoutStride>("LayoutStride");

    // FixedMultiIndexSet
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("FixedMultiIndexSet")
        .apply<FixedMultiIndexSet<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MaxDegreesExtent", [] (const WrappedT &set) { return set.MaxDegrees().extent(0); });
    });

    // ParameterizedFunctionBase
    mod.add_type<ParameterizedFunctionBase<Kokkos::HostSpace>>("ParameterizedFunctionBase")
        .method("CoeffMap" , &CoeffMapJulia)
        .method("SetCoeffs", &SetCoeffsJulia)
        .method("Evaluate" , &EvaluateJulia)
        .method("numCoeffs", &numCoeffsJulia)
        .method("inputDim" , &inputDimJulia)
        .method("outputDim", &outputDimJulia)
    ;

    // ConditionalMapBase
    mod.add_type<ConditionalMapBase<Kokkos::HostSpace>>("ConditionalMapBase")
        .method("CoeffMap", &CoeffMapJulia)
        .method("SetCoeffs", &SetCoeffsJulia)
        .method("Evaluate", &EvaluateJulia)
        .method("LogDeterminant", [](ConditionalMapBase<Kokkos::HostSpace>& map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
            map.LogDeterminantImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("GetBaseFunction", &ConditionalMapBase<Kokkos::HostSpace>::GetBaseFunction)
        .method("numCoeffs", &numCoeffsJulia)
        .method("inputDim" , &inputDimJulia)
        .method("outputDim", &outputDimJulia)
        // .method("Inverse", [](WrappedT &map, std::vector<double> x1, std::vector<double> r) {
        //             auto x1_sz = x1.size();
        //             int n_inp = map.inputDim;
        //             int n_pts = x1_sz / n_inp;
        //             auto view_x1 = ToKokkos(x1.data(), n_pts, n_inp);
        //             auto view_r = ToKokkos(r.data(), n_pts, n_inp);
        //             auto output = map.Inverse(view_x1, view_r);
        //             return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
        //         });
        // .method("LogDeterminantCoeffGrad", [](WrappedT &map, std::vector<double> pts) {
        //             auto sz = pts.size();
        //             int n_inp = map.inputDim;
        //             int n_pts = sz / n_inp;
        //             auto view = ToKokkos(pts.data(), n_pts, n_inp);
        //             auto output = map.LogDeterminantCoeffGrad(view);
        //             return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
        //         });
        ;

    mod.add_type<MultiIndexSet>("MultiIndexSet")
        .constructor<const unsigned int>()
        .method("Fix", &MultiIndexSet::Fix)
        .method("Union", &MultiIndexSet::Union)
        .method("IndexToMulti", &MultiIndexSet::IndexToMulti)
        .method("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .method("MaxOrders", &MultiIndexSet::MaxOrders)
        .method("AddActive", &MultiIndexSet::AddActive)
        .method("Frontier", &MultiIndexSet::Frontier)
        .method("Margin", &MultiIndexSet::Margin)
        .method("ReducedMargin", &MultiIndexSet::ReducedMargin)
        .method("StrictFrontier", &MultiIndexSet::StrictFrontier)
        .method("IsExpandable", &MultiIndexSet::IsExpandable)
        .method("NumActiveForward", &MultiIndexSet::NumActiveForward)
        .method("NumForward", &MultiIndexSet::NumForward)
    ;

    mod.method("MultiIndexSet", [](jlcxx::ArrayRef<int,2> idxs) {
        return MultiIndexSet(JuliaToEigen(idxs));
    });

    mod.set_override_module(jl_base_module);
    mod.method("sum", [](MultiIndex const& idx){ return idx.Sum(); });
    mod.method("setindex!", [](MultiIndex& idx, unsigned int val, unsigned int ind) { return idx.Set(ind, val); });
    // mod.method("setindex!", [](MultiIndexSet& idx, MultiIndex val, unsigned int ind) { return idx[ind] = val; });
    mod.method("getindex", [](MultiIndex const& idx, unsigned int ind) { return idx.Get(ind); });
    mod.method("maximum", [](MultiIndex const& idx){ return idx.Max(); });
    mod.method("String", [](MultiIndex const& idx){ return idx.String(); });
    mod.method("length", [](MultiIndex const& idx){ return idx.Length(); });
    mod.method("length", [](MultiIndexSet const& idx){ return idx.Length(); });
    mod.method("==", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 == idx2; });
    mod.method("!=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 != idx2; });
    mod.method("<", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 < idx2; });
    mod.method(">", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 > idx2; });
    mod.method("<=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 <= idx2; });
    mod.method(">=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 >= idx2; });
    mod.method("print", [](MultiIndex const& idx){std::cout << idx.String() << std::flush;});
    mod.unset_override_module();

    // BasisTypes
    mod.add_bits<BasisTypes>("BasisTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ProbabilistHermite", BasisTypes::ProbabilistHermite);
    mod.set_const("PhysicistHermite", BasisTypes::PhysicistHermite);
    mod.set_const("HermiteFunctions", BasisTypes::HermiteFunctions);

    // PosFuncTypes
    mod.add_bits<PosFuncTypes>("PosFuncTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("Exp", PosFuncTypes::Exp);
    mod.set_const("SoftPlus", PosFuncTypes::SoftPlus);

    // QuadTypes
    mod.add_bits<QuadTypes>("QuadTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ClenshawCurtis", QuadTypes::ClenshawCurtis);
    mod.set_const("AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    mod.set_const("AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);

    // MapOptions
    mod.add_type<MapOptions>("MapOptions").constructor<>()
       .method("BasisType!", [](MapOptions &opts, BasisTypes basis){ opts.basisType = basis; })
       .method("PosFuncType!", [](MapOptions &opts, PosFuncTypes f){ opts.posFuncType = f; })
       .method("QuadType!", [](MapOptions &opts, QuadTypes quad){ opts.quadType = quad; })
       .method("QuadAbsTol!", [](MapOptions &opts, double tol){ opts.quadAbsTol = tol; })
       .method("QuadRelTol!", [](MapOptions &opts, double tol){ opts.quadRelTol = tol; })
       .method("QuadMaxSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMaxSub = sub; })
       .method("QuadMinSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMinSub = sub; })
       .method("QuadPts!", [](MapOptions &opts, unsigned int pts){ opts.quadPts = pts; })
       .method("ContDeriv!", [](MapOptions &opts, bool deriv){ opts.contDeriv = deriv; })
       ;

    mod.add_type<TriangularMap<Kokkos::HostSpace>>("TriangularMap")
       .constructor<std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>>()
       .method("InverseInplace", [](TriangularMap<Kokkos::HostSpace> &map, jlcxx::Array<double,2> x, jlcxx::Array<double,2> r){
            map.InverseInplace(JuliaToKokkos(x), JuliaToKokkos(r));
       })
       .method("GetComponent", &TriangularMap<Kokkos::HostSpace>::GetComponent)
    ;
    // CreateComponent
    mod.method("CreateComponent", [](FixedMultiIndexSet<Kokkos::HostSpace> const & mset, MapOptions opts) {
        return MapFactory::CreateComponent(mset, opts);
    });
}