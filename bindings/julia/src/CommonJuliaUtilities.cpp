#include "CommonJuliaUtilities.h"

using namespace mpart::binding;

std::string mpart::binding::makeInitArguments(jlcxx::ArrayRef<char*> opts) {
    std::vector<std::string> args;

    for(int i = 0; i < opts.size(); i+=2){
        auto key = std::string(opts[i]);
        auto val = std::string(opts[i+1]);
        std::string opt = "--" + key + "=" + val;
        args.push_back(opt);
    }
    return args;
}

// Define a wrapper around Kokkos::Initialize that accepts an array of Julia CStrings as options
KokkosRuntime mpart::binding::KokkosInit(jlcxx::ArrayRef<char*> opts) {
    return KokkosRuntime(makeInitArguments(opts));
}

KokkosCustomPointer<KokkosRuntime> initKokkosRuntime(jlcxx::ArrayRef<char*> opts) {
    return KokkosCustomPointer<KokkosRuntime>(new KokkosRuntime(makeInitArguments(opts)));
}

void mpart::binding::CommonUtilitiesWrapper(jlcxx::Module &m)
{
    m.add_type<KokkosRuntime>(m, "KokkosRuntime")
        .constructor()
        .method("ElapsedTime", &KokkosRuntime::ElapsedTime);
    m.method("KokkosInit", &Kokkos::Init)
        .method("KokkosFinalize", &Kokkos::finalize);
}
