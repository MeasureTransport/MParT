#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "CommonJuliaUtilities.h"
#include "CommonUtilities.h"

using namespace mpart::binding;

void mpart::binding::MultiIndexWrapper(jlcxx::module &mod) {
    mod.add_type<MultiIndex>("MultiIndex")
        .constructor()
        .constructor<unsigned int, unsigned int>()
        .constructor<unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("NumNz", &MultiIndex::NumNz)
        .method("count_nonzero", &MultiIndex::NumNz);

    mod.add_type<MultiIndexSet>("MultiIndexSet")
        .constructor<const unsigned int>()
        .constructor<Eigen::Ref<const Eigen::MatrixXi> const&>(false)
        .method("fix", &MultiIndexSet::Fix)
        .method("CreateTotalOrder", &MultiIndexSet::CreateTotalOrder)
        .method("CreateTensorProduct", &MultiIndexSet::CreateTensorProduct)
        .method("union", &MultiIndexSet::Union)
        .method("SetLimiter", &MultiIndexSet::SetLimiter)
        .method("GetLimiter", &MultiIndexSet::GetLimiter)
        .method("IndexToMulti", &MultiIndexSet::IndexToMulti)
        .method("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .method("MaxOrders", &MultiIndexSet::MaxOrders)
        .method("Expand", &MultiIndexSet::Expand)
        .method("Activate", &MultiIndexSet::Activate)
        .method("AddActive", &MultiIndexSet::AddActive)
        .method("Frontier", &MultiIndexSet::Frontier)
        .method("Margin", &MultiIndexSet::Margin)
        .method("ReducedMargin", &MultiIndexSet::ReducedMargin)
        .method("StrictFrontier", &MultiIndexSet::StrictFrontier)
        .method("IsExpandable", &MultiIndexSet::IsExpandable)
        .method("NumActiveForward", &MultiIndexSet::NumActiveForward)
        .method("NumForward", &MultiIndexSet::NumForward)
    ;

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
    mod.add_type<MultiIndexLimiter::None>("None")
        .constructor()
        .method(&MultiIndexLimiter::None::operator())
    ;

    // And
    mod.add_type<MultiIndexLimiter::And>("And")
        .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>()
        .method(&MultiIndexLimiter::And::operator())
    ;

    // Or
    mod.add_type<MultiIndexLimiter::Or>("Or")
        .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>()
        .method(&MultiIndexLimiter::Or::operator())
    ;

    // Xor
    mod.add_type<MultiIndexLimiter::Xor>("Xor")
        .constructor<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>()
        .method(&MultiIndexLimiter::Xor::operator())
    ;

    mod.set_override_module(jl_base_module);
    mod.method("sum", [](MultiIndex const& idx){ return idx.Sum(); });
    mod.method("setindex!", [](MultiIndex& idx, unsigned int val, unsigned int ind) { return idx.Set(ind, val); });
    mod.method("setindex!", [](MultiIndexSet& idx, unsigned int val, unsigned int ind) { return idx[ind] = val; });
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