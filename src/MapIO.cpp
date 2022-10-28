#include "MapIO.h"

template<typename MemorySpace>
int SaveMapInfo(std::string const& filename, MapOptions const& options,
            FixedMultiIndexSet<MemorySpace> const& mset, StridedVector<double, MemorySpace> const& coeffs) {
    std::ofstream ofs(filename);
    if(!ofs.is_open())
        return -1;

    ofs << options << mset << vec;
    ofs.close();
    return 0;
}

template<typename MemorySpace>
int LoadMapInfo(std::string const& filename, MapOptions& options,
            FixedMultiIndexSet<MemorySpace>& mset, StridedVector<double,MemorySpace>& coeffs) {
    std::ifstream ifs(filename);
    if(!ifs.is_open())
        return -1;

    ifs >> options >> mset >> vec;
    ifs.close();
    return 0;
}

