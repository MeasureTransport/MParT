#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/MapOptions.h"

#include "CommonJuliaUtilities.h"
#include "CommonUtilities.h"

#include <MParT/MapOptions.h>
#include <Kokkos_Core.hpp>

#include "CommonJuliaUtilities.h"

namespace jlcxx {
  template<> struct IsMirroredType<mpart::MultiIndexLimiter::None> : std::false_type { };
}

using namespace mpart;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    // CommonUtilitiesWrapper(mod);
    // MultiIndexWrapper(mod);
    // MapOptionsWrapper(mod);

    mod.method("Initialize", [](jlcxx::ArrayRef<char*> opts){mpart::binding::Initialize(opts);});

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
        wrapped.template constructor<unsigned int, unsigned int>();
        wrapped.method("MaxDegreesExtent", [] (const WrappedT &set) { return set.MaxDegrees().extent(0); });
    });

    mod.add_type<MultiIndexSet>("MultiIndexSet")
        .constructor<const unsigned int>()
        .constructor<jlcxx::ConstArray<unsigned int,2> const&>()
        .method("fix", &MultiIndexSet::Fix)
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

    mod.method("MultiIndexSetConstruct", [](jlcxx::ConstArray<int,2> const& idxs) {
        auto sz = idxs.size();
        return MultiIndexSet(Eigen::Map<const Eigen::MatrixXi>(idxs.ptr(), std::get<0>(sz)*std::get<1>(sz)));
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
}