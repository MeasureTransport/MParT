#include "CommonJuliaUtilities.h"

JLCXX_MODULE mpart::julia::MultiIndexWrapper(jlcxx::Module& mod)
{
    mod.add_type<MultiIndex>("MultiIndex")
        .constructor<>()
        .constructor<unsigned int, unsigned int>()
        .constructor<std::vector<unsigned int> const&>()
        .method("sum", &MultiIndex::sum)
        .method("max", &MultiIndex::max)
        .method("count_nonzero", &MultiIndex::NumNz);
    mod.set_override_module(mod.julia_module());
    mod.method("setindex", &MultiIndex::Set)
       .method("getindex", &MultiIndex::Get)
       .method("String", &MultiIndex::String)
       .method("length", &MultiIndex::Length)
       .method("==", &MultiIndex::operator==)
       .method("!=", &MultiIndex::operator!=)
       .method("<", &MultiIndex::operator<)
       .method(">", &MultiIndex::operator>)
       .method("<=", &MultiIndex::operator<=)
       .method(">=", &MultiIndex::operator>=);
    mod.unset_override_module();
}