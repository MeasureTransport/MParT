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
    // template<> struct SuperType<mpart::ConditionalMapBase<Kokkos::HostSpace>> {typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
    
    
}

namespace mpart {
    // struct ParameterizedFunctionBaseHost {
    //     pfb::ParameterizedFunctionBase<Kokkos::HostSpace>;
    // };

    // struct ConditionalMapBaseHost {
    //     cmb::ConditionalMapBase<Kokkos::HostSpace>;
    // };

    // using PFBH = ParameterizedFunctionBase<Kokkos::HostSpace>;
    // using CMBH = ConditionalMapBase<Kokkos::HostSpace>;

    // struct PFBH {
    //     ParameterizedFunctionBase<Kokkos::HostSpace> pfb;
    // };

    // struct CMBH {
    //     ConditionalMapBase<Kokkos::HostSpace> cmb;
    // }

    jlcxx::ArrayRef<double> evaluateFcn(ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, StridedMatrix<double, Kokkos::HostSpace> &pts) {
        auto sz = pts.size();
        int n_inp = pfb.inputDim;
        int n_pts = sz / n_inp;
        auto out_vec = pfb.Evaluate(pts);
        return jlcxx::make_julia_array(out_vec.data(), out_vec.size());
    }
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

    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>,jlcxx::TypeVar<2>>>("View")
        .apply<StridedVector<double, Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("VecToMParT", &VecToKokkos<double,Kokkos::HostSpace>);
        }).apply<StridedMatrix<double, Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MatToMParT", &MatToKokkos<double,Kokkos::HostSpace>);
        }).apply<StridedVector<const double, Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("VecToConstMParT", &VecToConstKokkos<double,Kokkos::HostSpace>);
        }).apply<StridedMatrix<const double, Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MatToConstMParT", &MatToConstKokkos<double,Kokkos::HostSpace>);
        });

    // FixedMultiIndexSet
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("FixedMultiIndexSet")
        .apply<FixedMultiIndexSet<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MaxDegreesExtent", [] (const WrappedT &set) { return set.MaxDegrees().extent(0); });
    });

    // ParameterizedFunctionBase
    // mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("ParameterizedFunctionBase")
    //     .apply<ParameterizedFunctionBase<Kokkos::HostSpace>>([](auto wrapped) {
    //         typedef typename decltype(wrapped)::type WrappedT;
    //         // typedef typename WrappedT::fake_val fake_val;
    //         wrapped.method("CoeffMap", [](WrappedT &w){ return KokkosToJulia(w.Coeffs()); });
    //         wrapped.method("SetCoeffs", [](WrappedT &w, jlcxx::ArrayRef<double> &v){ w.SetCoeffs(JuliaToKokkos(v)); });
    //         wrapped.method("Evaluate", [](WrappedT &w, jlcxx::ArrayRef<double,2> &pts){ return KokkosToJulia(w.Evaluate(JuliaToKokkos(pts))); });
    //         // wrapped.method("CoeffGrad", [](WrappedT& pfb, std::vector<double> pts, std::vector<double> sens) {
    //         //     auto sz = pts.size();
    //         //     int n_inp = pfb.inputDim;
    //         //     int n_pts = sz / n_inp;
    //         //     Kokkos::View<const double**, Kokkos::HostSpace> view_pts = ToKokkos(pts.data(), n_pts, n_inp);
    //         //     Kokkos::View<const double**, Kokkos::HostSpace> view_sens = ToKokkos(sens.data(), n_pts, n_inp);
    //         //     auto output = pfb.CoeffGrad(view_pts, view_sens);
    //         //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
    //         // });
    //         wrapped.method("numCoeffs", [](WrappedT& pfb) { return pfb.numCoeffs; });
    //         wrapped.method("inputDim", [](WrappedT& pfb) { return pfb.inputDim; });
    //         wrapped.method("outputDim", [](WrappedT& pfb) { return pfb.outputDim; });
    //     });

    // When CMBH and PFBH are just `using` statements
    // mod.add_type<PFBH>("ParameterizedFunctionBase")
    //     .method("CoeffMap", [](PFBH &f){ return KokkosToJulia(f.Coeffs()); })
    //     .method("SetCoeffs", [](PFBH &f, jlcxx::ArrayRef<double> &v){ f.SetCoeffs(JuliaToKokkos(v)); })
    //     .method("Evaluate", [](PFBH &f, jlcxx::ArrayRef<double,2> &pts){ return KokkosToJulia(f.Evaluate(JuliaToKokkos(pts))); })
    // ;
    // mod.add_type<CMBH>("ConditionalMapBase")
    //     .method("LogDeterminant", [](CMBH &map, jlcxx::ArrayRef<double,2> &pts){ return KokkosToJulia(map.LogDeterminant(JuliaToKokkos(pts))); })
    //     .method("to_base", [](std::shared_ptr<CMBH> &ptr){ return std::static_pointer_cast<ParameterizedFunctionBase<Kokkos::HostSpace>>(ptr); })
    // ;

    // ParameterizedFunctionBase
    mod.add_type<ParameterizedFunctionBase<Kokkos::HostSpace>>("ParameterizedFunctionBase")
        .method("CoeffMap", [](ParameterizedFunctionBase<Kokkos::HostSpace> &w){ return KokkosToJulia(w.Coeffs()); })
        .method("SetCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace> &w, jlcxx::ArrayRef<double> v){ w.SetCoeffs(JuliaToKokkos(v)); })
        .method("Evaluate", [](ParameterizedFunctionBase<Kokkos::HostSpace> &w, jlcxx::ArrayRef<double,2> pts){ return KokkosToJulia(w.Evaluate(JuliaToKokkos(pts))); })
        .method("numCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.numCoeffs; })
        .method("inputDim", [](ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.inputDim; })
        .method("outputDim", [](ParameterizedFunctionBase<Kokkos::HostSpace>& pfb) { return pfb.outputDim; });

    mod.add_type<ConditionalMapBase<Kokkos::HostSpace>>("ConditionalMapBase")
        .method("CoeffMap", [](ConditionalMapBase<Kokkos::HostSpace> &w){ return KokkosToJulia(w.Coeffs()); })
        .method("SetCoeffs", [](ConditionalMapBase<Kokkos::HostSpace> &w, jlcxx::ArrayRef<double> v){ w.SetCoeffs(JuliaToKokkos(v)); })
        .method("Evaluate", [](ConditionalMapBase<Kokkos::HostSpace> &map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            unsigned int outDim = map.outputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outDim, numPts);
            map.EvaluateImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("LogDeterminant", [](ConditionalMapBase<Kokkos::HostSpace>& map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
            map.LogDeterminantImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("GetBaseFunction", &ConditionalMapBase<Kokkos::HostSpace>::GetBaseFunction)
        .method("numCoeffs", [](ConditionalMapBase<Kokkos::HostSpace>& pfb) { return pfb.numCoeffs; })
        .method("inputDim", [](ConditionalMapBase<Kokkos::HostSpace>& pfb) { return pfb.inputDim; })
        .method("outputDim", [](ConditionalMapBase<Kokkos::HostSpace>& pfb) { return pfb.outputDim; });
    
    // ConditionalMapBase
    // mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("ConditionalMapBase")
    //     .apply<ConditionalMapBase<Kokkos::HostSpace>>([](auto wrapped) {
    //         typedef typename decltype(wrapped)::type WrappedT;
    //         // typedef typename WrappedT::fake_value fake_value;
    //         wrapped.method("LogDeterminant", [](WrappedT& cmb, jlcxx::ArrayRef<double,2> &pts){ return KokkosToJulia(cmb.LogDeterminant(JuliaToKokkos(pts))); });
    //         wrapped.method("to_base", [] (std::shared_ptr<WrappedT> w) { return std::static_pointer_cast<ParameterizedFunctionBase<Kokkos::HostSpace>>(w); });
    //         // wrapped.method("Inverse", [](WrappedT &map, std::vector<double> x1, std::vector<double> r) {
    //         //     auto x1_sz = x1.size();
    //         //     int n_inp = map.inputDim;
    //         //     int n_pts = x1_sz / n_inp;
    //         //     auto view_x1 = ToKokkos(x1.data(), n_pts, n_inp);
    //         //     auto view_r = ToKokkos(r.data(), n_pts, n_inp);
    //         //     auto output = map.Inverse(view_x1, view_r);
    //         //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
    //         // });
    //         // wrapped.method("LogDeterminantCoeffGrad", [](WrappedT &map, std::vector<double> pts) {
    //         //     auto sz = pts.size();
    //         //     int n_inp = map.inputDim;
    //         //     int n_pts = sz / n_inp;
    //         //     auto view = ToKokkos(pts.data(), n_pts, n_inp);
    //         //     auto output = map.LogDeterminantCoeffGrad(view);
    //         //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
    //         // });
    //         wrapped.method("GetBaseFunction", &WrappedT::GetBaseFunction);
    // });

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
    mod.add_type<MapOptions>("MapOptions").constructor<>();

    // CreateComponent
    mod.method("CreateComponent", [](FixedMultiIndexSet<Kokkos::HostSpace> const & mset, MapOptions opts) {
        return MapFactory::CreateComponent(mset, opts);
    });
}