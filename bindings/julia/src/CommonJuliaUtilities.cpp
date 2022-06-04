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

// // Define a wrapper around Kokkos::Initialize that accepts an array of Julia CStrings as options
// KokkosRuntime mpart::binding::KokkosInit(jlcxx::ArrayRef<char*> opts) {
//     return KokkosInit(makeInitArguments(opts));
// }

KokkosCustomPointer<KokkosRuntime> initKokkosRuntime(jlcxx::ArrayRef<char*> opts) {
    KokkosInit(makeInitArguments(opts));
    return KokkosCustomPointer<KokkosRuntime>(new KokkosRuntime());
}

void mpart::binding::CommonUtilitiesWrapper(jlcxx::Module &m)
{
    m.add_type<KokkosRuntime>("KokkosRuntime")
        .constructor<>()
        .method("ElapsedTime", &KokkosRuntime::ElapsedTime);
    m.method("KokkosInit", &initKokkosRuntime);
    //m.method("KokkosFinalize", &Kokkos::finalize);
}
