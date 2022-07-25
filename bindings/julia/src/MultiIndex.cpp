#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "JlArrayConversions.h"
#include "CommonJuliaUtilities.h"

namespace jlcxx {
    // This fixes a Compilation error that I don't understand
    template<> struct IsMirroredType<mpart::MultiIndexLimiter::None> : std::false_type { };
}

using namespace mpart::binding;

void mpart::binding::MultiIndexWrapper(jlcxx::Module &mod) {
    mod.add_type<MultiIndex>("MultiIndex")
        .constructor()
        .constructor<unsigned int, unsigned int>()
        .constructor<unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("NumNz", &MultiIndex::NumNz)
        .method("count_nonzero", &MultiIndex::NumNz);

    jlcxx::stl::apply_stl<MultiIndex>(mod);

    // FixedMultiIndexSet
    mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("FixedMultiIndexSet")
        .apply<FixedMultiIndexSet<Kokkos::HostSpace>>([](auto wrapped) {
            typedef typename decltype(wrapped)::type WrappedT;
            wrapped.method("MaxDegreesExtent", [] (const WrappedT &set) { return set.MaxDegrees().extent(0); });
    });

    // MultiIndexSet
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
}