#include "CommonJuliaUtilities.h"

#include <Kokkos_Core.hpp>

using namespace mpart::binding;

namespace mpart{
    namespace binding{
        std::vector<std::string> makeInitArguments(std::vector<std::string> opts) {
            std::vector<std::string> args;

            for(int i = 0; i < opts.size(); i+=2){
                auto key = opts[i];
                auto val = opts[i+1];
                std::string opt = "--" + key + "=" + val;
                args.push_back(opt);
            }
            return args;
        }
    }
}

void mpart::binding::CommonUtilitiesWrapper(jlcxx::Module &mod)
{
    mod.method("Initialize", [](){mpart::binding::Initialize(std::vector<std::string> {});});
    mod.method("Initialize", [](std::vector<std::string> v){mpart::binding::Initialize(makeInitArguments(v));});
    mod.add_type<Kokkos::HostSpace>("HostSpace");
    mod.add_type<Kokkos::LayoutStride>("LayoutStride");
    mod.method("Concurrency", &Kokkos::DefaultExecutionSpace::concurrency);
}
