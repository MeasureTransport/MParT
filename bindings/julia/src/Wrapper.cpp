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

#include "CommonJuliaUtilities.h"

namespace jlcxx {
  template<> struct IsMirroredType<mpart::MultiIndexLimiter::None> : std::false_type { };
  template<> struct SuperType<mpart::ConditionalMapBase<Kokkos::HostSpace>> {typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
}
namespace mpart {
    struct FixedMultiIndexSetHost {
        FixedMultiIndexSetHost(FixedMultiIndexSet<Kokkos::HostSpace> const& set): mset(set) {}
        FixedMultiIndexSet<Kokkos::HostSpace> const& mset;
    };

    struct ConditionalMapBaseHost {
        ConditionalMapBaseHost(ConditionalMapBase<Kokkos::HostSpace> const& map): mmap(map) {}
        ConditionalMapBase<Kokkos::HostSpace> const& mmap;
    };

    std::vector<double> logDetFcn(ConditionalMapBase<Kokkos::HostSpace> &map, std::vector<double> pts) {
        auto sz = pts.size();
        int n_inp = map.inputDim;
        int n_pts = sz / n_inp;
        auto view = ToConstKokkos(pts.data(), n_inp, n_pts);
        // typedef typename decltype(view)::fake_value fake_value;
        Kokkos::View<double*, Kokkos::HostSpace> out_view("Log Determinants", view.extent(1));
        map.LogDeterminantImpl(view, out_view);
        return std::vector<double>(out_view.data(), out_view.data() + out_view.extent(0));
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
    // FixedMultiIndexSet
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("FixedMultiIndexSet")
        .apply<FixedMultiIndexSet<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MaxDegreesExtent", [] (const WrappedT &set) { return set.MaxDegrees().extent(0); });
    });

    // ParameterizedFunctionBase
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("ParameterizedFunctionBase")
        .apply<ParameterizedFunctionBase<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("CoeffMap", [](WrappedT& pfb) {
                auto view = pfb.Coeffs();
                return jlcxx::make_julia_array(view.data(), view.extent(0));
            });
            wrapped.method("SetCoeffs", [](WrappedT& pfb, std::vector<double> coeffs) {
                auto view = ToKokkos(coeffs.data(), coeffs.size());
                pfb.SetCoeffs(view);
            });
            // wrapped.method("Evaluate", [](WrappedT& pfb, std::vector<double> pts) {
            //     auto sz = pts.size();
            //     int n_inp = pfb.inputDim;
            //     int n_pts = sz / n_inp;
            //     Kokkos::View<const double**, Kokkos::HostSpace> view = ToConstKokkos(pts.data(), n_pts, n_inp);
            //     typedef typename decltype(view)::fake_value fake_value;
            //     auto output = pfb.Evaluate(view);
            //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
            // });
            // wrapped.method("CoeffGrad", [](WrappedT& pfb, std::vector<double> pts, std::vector<double> sens) {
            //     auto sz = pts.size();
            //     int n_inp = pfb.inputDim;
            //     int n_pts = sz / n_inp;
            //     Kokkos::View<const double**, Kokkos::HostSpace> view_pts = ToKokkos(pts.data(), n_pts, n_inp);
            //     Kokkos::View<const double**, Kokkos::HostSpace> view_sens = ToKokkos(sens.data(), n_pts, n_inp);
            //     auto output = pfb.CoeffGrad(view_pts, view_sens);
            //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
            // });
            wrapped.method("numCoeffs", [](WrappedT& pfb) { return pfb.numCoeffs; });
            wrapped.method("inputDim", [](WrappedT& pfb) { return pfb.inputDim; });
            wrapped.method("outputDim", [](WrappedT& pfb) { return pfb.outputDim; });
        });

    // ConditionalMapBase
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("ConditionalMapBase")
        .apply<ConditionalMapBase<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            // typedef typename WrappedT::fake_value fake_value;
            // wrapped.method("LogDeterminant", &logDetFcn);
            // wrapped.method("LogDeterminant", [](WrappedT &map, std::vector<double> pts) {
            //     auto sz = pts.size();
            //     int n_inp = map.inputDim;
            //     int n_pts = sz / n_inp;
            //     auto view = ToConstKokkos(pts.data(), n_inp, n_pts);
            //     // typedef typename decltype(view)::fake_value fake_value;
            //     Kokkos::View<double*, Kokkos::HostSpace> out_view("Log Determinants", view.extent(1));
            //     map.LogDeterminantImpl(view, out_view);
            //     return std::vector<double>(out_view.data(), out_view.data() + out_view.extent(0));
            // });
            // wrapped.method("Inverse", [](WrappedT &map, std::vector<double> x1, std::vector<double> r) {
            //     auto x1_sz = x1.size();
            //     int n_inp = map.inputDim;
            //     int n_pts = x1_sz / n_inp;
            //     auto view_x1 = ToKokkos(x1.data(), n_pts, n_inp);
            //     auto view_r = ToKokkos(r.data(), n_pts, n_inp);
            //     auto output = map.Inverse(view_x1, view_r);
            //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
            // });
            // wrapped.method("LogDeterminantCoeffGrad", [](WrappedT &map, std::vector<double> pts) {
            //     auto sz = pts.size();
            //     int n_inp = map.inputDim;
            //     int n_pts = sz / n_inp;
            //     auto view = ToKokkos(pts.data(), n_pts, n_inp);
            //     auto output = map.LogDeterminantCoeffGrad(view);
            //     return jlcxx::make_julia_array(output.data(), output.extent(0), output.extent(1));
            // });
            wrapped.method("GetBaseFunction", &WrappedT::GetBaseFunction);
    });

    mod.add_type<MultiIndexSet>("MultiIndexSet")
        .constructor<const unsigned int>()
        .method("fix", [](MultiIndexSet mset) {return FixedMultiIndexSetHost(mset.Fix()); })
        // .method("CreateTotalOrder", &MultiIndexSet::CreateTotalOrder)
        // .method("CreateTensorProduct", &MultiIndexSet::CreateTensorProduct)
        .method("union", &MultiIndexSet::Union)
        // .method("SetLimiter", &MultiIndexSet::SetLimiter)
        // .method("GetLimiter", &MultiIndexSet::GetLimiter)
        .method("IndexToMulti", &MultiIndexSet::IndexToMulti)
        .method("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .method("MaxOrders", &MultiIndexSet::MaxOrders)

        // .method("Expand", &MultiIndexSet::Expand)
        // .method("Activate", &MultiIndexSet::Activate)

        .method("AddActive", &MultiIndexSet::AddActive)
        .method("Frontier", &MultiIndexSet::Frontier)
        .method("Margin", &MultiIndexSet::Margin)
        .method("ReducedMargin", &MultiIndexSet::ReducedMargin)
        .method("StrictFrontier", &MultiIndexSet::StrictFrontier)
        .method("IsExpandable", &MultiIndexSet::IsExpandable)
        .method("NumActiveForward", &MultiIndexSet::NumActiveForward)
        .method("NumForward", &MultiIndexSet::NumForward)
    ;

    mod.method("MultiIndexSet", [](std::vector<int>& idxs, unsigned int sz0, unsigned int sz1) {
        auto ptr = idxs.data();
        return MultiIndexSet(Eigen::Map<const Eigen::Matrix<int,Eigen::Dynamic,1>, 0, Eigen::OuterStride<>>(ptr, sz0, sz1, Eigen::OuterStride<>(std::max(sz0, sz1))));
    });

    // MultiIndexSetLimiters
    // TotalOrder
    mod.add_type<MultiIndexLimiter::TotalOrder>("TotalOrder")
        .constructor<unsigned int>()
        .method(&MultiIndexLimiter::TotalOrder::operator())
    ;

    // Dimension
    mod.add_type<MultiIndexLimiter::Dimension>("Dimension")
        .constructor<unsigned int, unsigned int>()
        .method(&MultiIndexLimiter::Dimension::operator())
    ;

    // Anisotropic
    mod.add_type<MultiIndexLimiter::Anisotropic>("Anisotropic")
        .constructor<std::vector<double> const&, double>()
        .method(&MultiIndexLimiter::Anisotropic::operator())
    ;

    // MaxDegree
    mod.add_type<MultiIndexLimiter::MaxDegree>("MaxDegree")
        .constructor<unsigned int, unsigned int>()
        .method(&MultiIndexLimiter::MaxDegree::operator())
    ;

    // None
    mod.add_type<MultiIndexLimiter::None>("NoneLim")
        .constructor<>()
        .method(&MultiIndexLimiter::None::operator())
    ;

    // And
    // mod.add_type<MultiIndexLimiter::And>("And")
    //     .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>()
    //     .method(&MultiIndexLimiter::And::operator())
    // ;

    // Or
    // mod.add_type<MultiIndexLimiter::Or>("Or")
    //     .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>()
    //     .method(&MultiIndexLimiter::Or::operator())
    // ;

    // Xor
    // mod.add_type<MultiIndexLimiter::Xor>("Xor")
    //     .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>()
    //     .method(&MultiIndexLimiter::Xor::operator())
    // ;

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