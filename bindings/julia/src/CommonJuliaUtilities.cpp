#include "CommonJuliaUtilities.h"

#include <Kokkos_Core.hpp>

using namespace mpart::binding;

namespace mpart{
    namespace binding{
        std::vector<std::string> makeInitArguments(jlcxx::ArrayRef<char*> opts) {
            std::vector<std::string> args;

            for(int i = 0; i < opts.size(); i+=2){
                auto key = std::string(opts[i]);
                auto val = std::string(opts[i+1]);
                std::string opt = "--" + key + "=" + val;
                args.push_back(opt);
            }
            return args;
        }
    }
}


void mpart::binding::Initialize(jlcxx::ArrayRef<char*> opts) {
    mpart::binding::Initialize(makeInitArguments(opts));
}

void mpart::binding::CommonUtilitiesWrapper(jlcxx::Module &mod)
{
    mod.method("Initialize", [](){mpart::binding::Initialize(std::vector<std::string> {});});
    mod.add_type<Kokkos::HostSpace>("HostSpace");
    mod.add_type<Kokkos::LayoutStride>("LayoutStride");
}
