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


void Initialize(jlcxx::ArrayRef<char*> opts) {
    mpart::binding::Initialize(makeInitArguments(opts));
}

void mpart::binding::CommonUtilitiesWrapper(jlcxx::Module &mod)
{
    m.method("Initialize", &Initialize);
}
