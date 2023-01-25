#include <fstream>
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

    mod.add_type<FixedMultiIndexSet<Kokkos::HostSpace>>("FixedMultiIndexSet")
        .constructor<unsigned int, unsigned int>()
        .method("MaxDegreesExtent", [] (const FixedMultiIndexSet<Kokkos::HostSpace> &set) { return set.MaxDegrees().extent(0); })
        .method("Serialize", [](const FixedMultiIndexSet<Kokkos::HostSpace> &set, std::string &filename){
#if defined(MPART_HAS_CEREAL)
            std::ofstream os (filename);
            cereal::BinaryOutputArchive oarchive(os);
            oarchive(set);
#else
            std::cerr << "FixedMultiIndexSet::Serialize: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif
        })
        .method("Deserialize", [](FixedMultiIndexSet<Kokkos::HostSpace> &set, std::string &filename){
#if defined(MPART_HAS_CEREAL)
            std::ifstream is (filename);
            cereal::BinaryInputArchive iarchive(is);
            iarchive(set);
#else
            std::cerr << "FixedMultiIndexSet::Deserialize: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif
        })
    ;

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
        .method("Size", &MultiIndexSet::Size)
    ;

    mod.method("MultiIndexSet", [](jlcxx::ArrayRef<int,2> idxs) {
        return MultiIndexSet(JuliaToEigenMat(idxs));
    });

    mod.method("CreateTotalOrder", [](unsigned int length, unsigned int maxOrder){return MultiIndexSet::CreateTotalOrder(length, maxOrder, MultiIndexLimiter::None()); });

    mod.set_override_module(jl_base_module);
    mod.method("sum", [](MultiIndex const& idx){ return idx.Sum(); });
    mod.method("setindex!", [](MultiIndex& idx, unsigned int val, unsigned int ind) { return idx.Set(ind-1, val); });
    mod.method("getindex", [](MultiIndex const& idx, unsigned int ind) { return idx.Get(ind-1); });
    mod.method("getindex", [](MultiIndexSet const& idx, int ind) { return idx.at(ind-1); });
    mod.method("maximum", [](MultiIndex const& idx){ return idx.Max(); });
    mod.method("string", [](MultiIndex const& idx){ return idx.String(); });
    mod.method("length", [](MultiIndex const& idx){ return idx.Length(); });
    mod.method("length", [](MultiIndexSet const& idx){ return idx.Length(); });
    mod.method("length", [](FixedMultiIndexSet<Kokkos::HostSpace> &mset){return mset.Length();});
    mod.method("size", [](FixedMultiIndexSet<Kokkos::HostSpace> &mset){return mset.Size();});
    mod.method("vec", [](MultiIndex const& idx){ return idx.Vector(); });
    mod.method("==", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 == idx2; });
    mod.method("!=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 != idx2; });
    mod.method("<", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 < idx2; });
    mod.method(">", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 > idx2; });
    mod.method("<=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 <= idx2; });
    mod.method(">=", [](MultiIndex const& idx1, MultiIndex const& idx2){ return idx1 >= idx2; });
    mod.method("print", [](MultiIndex const& idx){std::cout << idx.String() << std::flush;});
    mod.unset_override_module();
}