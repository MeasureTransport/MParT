#include "CommonJuliaUtilities.h"
#include "MParT/MultiIndices/MultiIndex.h"

using namespace mpart;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.add_type<MultiIndex>("MultiIndex")
        .constructor()
        .constructor<unsigned int, unsigned int>()
        .constructor<unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("count_nonzero", &MultiIndex::NumNz);
    mod.set_override_module(jl_base_module);
    mod.method("sum", [](MultiIndex idx){ return idx.Sum(); });
    mod.method("setindex!", [](MultiIndex idx, unsigned int val, unsigned int ind) { return idx.Set(ind, val); });
    mod.method("getindex", [](MultiIndex idx, unsigned int ind) { return idx.Get(ind); });
    mod.method("maximum", [](MultiIndex idx){ return idx.Max(); });
    mod.method("String", [](MultiIndex idx){ return idx.String().c_str(); });
    mod.method("length", [](MultiIndex idx){ return idx.Length(); });
    mod.method("==", [](MultiIndex idx1, MultiIndex idx2){ return idx1 == idx2; });
    mod.method("!=", [](MultiIndex idx1, MultiIndex idx2){ return idx1 != idx2; });
    mod.method("<", [](MultiIndex idx1, MultiIndex idx2){ return idx1 < idx2; });
    mod.method(">", [](MultiIndex idx1, MultiIndex idx2){ return idx1 > idx2; });
    mod.method("<=", [](MultiIndex idx1, MultiIndex idx2){ return idx1 <= idx2; });
    mod.method(">=", [](MultiIndex idx1, MultiIndex idx2){ return idx1 >= idx2; });
    mod.unset_override_module();
}