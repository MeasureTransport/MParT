#include "CommonJuliaUtilities.h"
#include "MParT/MultiIndices/MultiIndex.h"

using namespace mpart;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.add_type<MultiIndex>("MultiIndex")
        .constructor()
        .constructor<unsigned int, unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("count_nonzero", &MultiIndex::NumNz);
    mod.set_override_module(jl_base_module);
    mod.method("sum", [](MultiIndex idx){ return idx.Sum(); });
    //    .method("max", &MultiIndex::Max)
    //    .method("setindex", &MultiIndex::Set)
    //    .method("getindex", &MultiIndex::Get)
    //    .method("String", &MultiIndex::String)
    //    .method("length", &MultiIndex::Length)
    //    .method("==", &MultiIndex::operator==)
    //    .method("!=", &MultiIndex::operator!=)
    //    .method("<", &MultiIndex::operator<)
    //    .method(">", &MultiIndex::operator>)
    //    .method("<=", &MultiIndex::operator<=)
    //    .method(">=", &MultiIndex::operator>=);
    mod.unset_override_module();
}