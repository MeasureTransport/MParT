#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include "MParT/Utilities/ArrayConversions.h"
#include <stdio.h>

using namespace mpart;


namespace mpart{

    template<typename MemorySpace>
    struct StartSetter {

        StartSetter(Kokkos::View<unsigned int*, MemorySpace> nzStarts,
                   unsigned int dim) : _nzStarts(nzStarts), _dim(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            this->_nzStarts(i) = i*_dim;
        };

        Kokkos::View<unsigned int*, MemorySpace> _nzStarts;
        const unsigned int _dim;
    };

    template<typename MemorySpace>
    struct DimSetter {

        DimSetter(Kokkos::View<unsigned int*, MemorySpace> nzDims,
                  unsigned int dim) : _nzDims(nzDims), _dim(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            this->_nzDims(i) = i%_dim;
        };

        Kokkos::View<unsigned int*, MemorySpace> _nzDims;
        const unsigned int _dim;
    };

    template<typename MemorySpace>
    struct MaxDegreeSetter {

        MaxDegreeSetter(Kokkos::View<unsigned int*, MemorySpace> maxDegrees,
                        Kokkos::View<unsigned int*, MemorySpace> nzDims,
                        Kokkos::View<unsigned int*, MemorySpace> nzOrders,
                        unsigned int dim) : maxDegrees_(maxDegrees), nzDims_(nzDims), nzOrders_(nzOrders), dim_(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            Kokkos::atomic_max(&maxDegrees_(nzDims_(i)), nzOrders_(i));
        }

        Kokkos::View<unsigned int*, MemorySpace> maxDegrees_;
        Kokkos::View<unsigned int*, MemorySpace> nzDims_;
        Kokkos::View<unsigned int*, MemorySpace> nzOrders_;
        const unsigned int dim_;
    };


    template<typename MemorySpace>
    struct MaxDegreeInitializer {

        MaxDegreeInitializer(Kokkos::View<unsigned int*, MemorySpace> maxDegrees) : maxDegrees_(maxDegrees){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            maxDegrees_(i) = 0;
        };

        Kokkos::View<unsigned int*, MemorySpace> maxDegrees_;
    };
}




template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int                dimen,
                                       Kokkos::View<unsigned int*, MemorySpace> nonzeroOrders) : dim(dimen),
                                                                                            isCompressed(false),
                                                                                            nzDims("Nonzero dims", nonzeroOrders.extent(0)),
                                                                                            nzOrders(nonzeroOrders)
{
    SetupTerms();
    CalculateMaxDegrees();
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::SetupTerms()
{

    unsigned int numTerms = nzOrders.extent(0) / dim;

    nzStarts = Kokkos::View<unsigned int*, MemorySpace>("Start of a Multiindex", numTerms+1);
    Kokkos::parallel_for(numTerms, StartSetter<MemorySpace>(nzStarts,dim));
    Kokkos::parallel_for(dim*numTerms, DimSetter<MemorySpace>(nzDims,dim));
}
template<>
void FixedMultiIndexSet<Kokkos::HostSpace>::SetupTerms()
{

    unsigned int numTerms = nzOrders.extent(0) / dim;

    nzStarts = Kokkos::View<unsigned int*, Kokkos::HostSpace>("Start of a Multiindex", numTerms+1);
    {
    StartSetter<Kokkos::HostSpace> functor(nzStarts,dim);
    for(unsigned int i=0; i<numTerms; ++i)
        functor(i);
    }
    {
    DimSetter<Kokkos::HostSpace> functor(nzDims,dim);
    for(unsigned int i=0; i<dim*numTerms; ++i)
        functor(i);
    }
}


template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::CalculateMaxDegrees()
{
    maxDegrees = Kokkos::View<unsigned int*, MemorySpace>("Maximum degrees", dim);

    Kokkos::parallel_for(dim, MaxDegreeInitializer<MemorySpace>(maxDegrees));
    Kokkos::parallel_for(nzOrders.extent(0), MaxDegreeSetter<MemorySpace>(maxDegrees, nzDims, nzOrders, dim));
}

template<>
void FixedMultiIndexSet<Kokkos::HostSpace>::CalculateMaxDegrees()
{
    maxDegrees = Kokkos::View<unsigned int*, Kokkos::HostSpace>("Maximum degrees", dim);

    {
    MaxDegreeInitializer<Kokkos::HostSpace> functor(maxDegrees);
    for(unsigned int i=0; i<dim; ++i)
        functor(i);
    }

    {
    MaxDegreeSetter<Kokkos::HostSpace> functor(maxDegrees, nzDims, nzOrders, dim);
    for(unsigned int i=0; i<nzOrders.extent(0); ++i)
        functor(i);
    }
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int                dim,
                                       Kokkos::View<unsigned int*, MemorySpace> nzStarts,
                                       Kokkos::View<unsigned int*, MemorySpace> nzDims,
                                       Kokkos::View<unsigned int*, MemorySpace> nzOrders) : dim(dim),
                                                                                isCompressed(true),
                                                                                nzStarts(nzStarts),
                                                                                nzDims(nzDims),
                                                                                nzOrders(nzOrders)
{
    CalculateMaxDegrees();
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int dim,
                                                    unsigned int maxOrder) : dim(dim), isCompressed(true)
{
    // Figure out the number of terms in the total order
    unsigned int numTerms, numNz;
    std::tie(numTerms,numNz) = TotalOrderSize(maxOrder, 0);

    // Allocate space for the multis in compressed form
    nzStarts = Kokkos::View<unsigned int*, MemorySpace>("nzStarts", numTerms+1);
    nzDims   = Kokkos::View<unsigned int*, MemorySpace>("nzDims", numNz);
    nzOrders = Kokkos::View<unsigned int*, MemorySpace>("nzOrders", numNz);

    // Compute the multis
    std::vector<unsigned int> workspace(dim);
    unsigned int currNz=0;
    unsigned int currTerm=0;

    FillTotalOrder(maxOrder, workspace, 0, currTerm, currNz);

    CalculateMaxDegrees();
}


template<typename MemorySpace>
Kokkos::View<const unsigned int*, MemorySpace> FixedMultiIndexSet<MemorySpace>::MaxDegrees() const
{
    return maxDegrees;
}

template<typename MemorySpace>
std::vector<unsigned int> FixedMultiIndexSet<MemorySpace>::IndexToMulti(unsigned int index) const
{
    assert(false);
    return std::vector<unsigned int>();
}

template<>
std::vector<unsigned int> FixedMultiIndexSet<Kokkos::HostSpace>::IndexToMulti(unsigned int index) const
{
    std::vector<unsigned int> output(dim,0);
    if(isCompressed){
        for(unsigned int i=nzStarts(index); i<nzStarts(index+1); ++i){
            output.at( nzDims(i) ) = nzOrders(i);
        }
    }else{
        for(unsigned int i=0; i<dim; ++i)
            output.at(i) = nzOrders(i + dim*index);
    }
    return output;
}


template<typename MemorySpace>
int FixedMultiIndexSet<MemorySpace>::MultiToIndex(std::vector<unsigned int> const& multi) const
{
    if(isCompressed){

        // Figure out how many nonzeros are in this multiindex
        unsigned int nnz = 0;
        for(auto& val : multi)
            nnz += (val>0) ? 1:0;

        // Now search for the matching multi
        for(unsigned int i=0; i<nzStarts.extent(0); ++i){

            // First, check if the number of nonzeros matches
            if((nzStarts(i+1)-nzStarts(i))==nnz){

                // Now check the contents
                bool matches = true;
                for(unsigned int j=nzStarts(i); j<nzStarts(i+1); ++j){
                    if(nzOrders(j)!=multi.at(nzDims(j))){
                        matches=false;
                        break;
                    }
                }

                // We found it!  Return the current index
                if(matches)
                    return i;
            }
        }

        // We didn't find it, return a negative value
        return -1;

    }else{
        unsigned int numTerms = Size();
        for(unsigned int i=0; i<numTerms; ++i){

            bool isMatch = true;
            for(unsigned int d=0; d<dim; ++d){
                if(multi.at(d) != nzOrders(d + i*dim)){
                    isMatch = false;
                    break;
                }
            }

            if(isMatch)
                return i;
        }

        return -1;

    }
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::Print() const
{
    if(isCompressed){
        std::cout << "Starts:\n";
        for(unsigned int i=0; i<nzStarts.extent(0); ++i)
            std::cout << nzStarts(i) << "  ";
        std::cout << std::endl;

        std::cout << "\nDims:\n";
        for(unsigned int i=0; i<nzDims.extent(0); ++i)
            std::cout << nzDims(i) << "  ";
        std::cout << std::endl;

        std::cout << "\nOrders:\n";
        for(unsigned int i=0; i<nzOrders.extent(0); ++i)
            std::cout << nzOrders(i) << "  ";
        std::cout << std::endl;
    }

    std::cout << "\nMultis:\n";
    std::vector<unsigned int> multi;
    for(unsigned int term=0; term<nzStarts.extent(0)-1; ++term){
        multi = IndexToMulti(term);

        for(auto& m : multi)
            std::cout << m << "  ";

        std::cout << std::endl;
    }

}

template<typename MemorySpace>
std::pair<unsigned int, unsigned int> FixedMultiIndexSet<MemorySpace>::TotalOrderSize(unsigned int maxOrder, unsigned int currDim)
{
    unsigned int numTerms=0;
    unsigned int numNz=0;
    unsigned int localTerms, localNz;
    if(currDim<dim-1) {
        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            std::tie(localTerms,localNz) = TotalOrderSize(maxOrder-pow,currDim+1);
            numTerms += localTerms;
            numNz += localNz + ((pow>0)?localTerms:0);
        }
    }else{
        numTerms = maxOrder+1;
        numNz = maxOrder;
    }

    return std::make_pair(numTerms, numNz);
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::FillTotalOrder(unsigned int maxOrder,
                                        std::vector<unsigned int> &workspace,
                                        unsigned int currDim,
                                        unsigned int &currTerm,
                                        unsigned int &currNz)
{

    if(currDim<dim-1) {
        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            workspace[currDim] = pow;
            FillTotalOrder(maxOrder-pow, workspace, currDim+1, currTerm, currNz);
        }
    }else{

        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            workspace[currDim] = pow;

            // Copy the multiindex into the compressed format
            nzStarts(currTerm) = currNz;
            for(unsigned int i=0; i<dim; ++i){
                if(workspace[i]>0){
                    nzDims(currNz) = i;
                    nzOrders(currNz) = workspace[i];
                    currNz++;
                }
            }

            currTerm++;
        }

    }

    if(currDim==0)
        nzStarts(currTerm) = currNz;
}

template<>
template<>
FixedMultiIndexSet<Kokkos::HostSpace> FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<Kokkos::HostSpace>()
{
    return *this;
}

// If a device is being used, compile code to copy the FixedMultiIndexSet to device memory
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template<>
    template<>
    FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<Kokkos::DefaultExecutionSpace::memory_space>()
    {
        auto deviceStarts = mpart::ToDevice<Kokkos::DefaultExecutionSpace::memory_space>(nzStarts);
        auto deviceDims = mpart::ToDevice<Kokkos::DefaultExecutionSpace::memory_space>(nzDims);
        auto deviceOrders =  mpart::ToDevice<Kokkos::DefaultExecutionSpace::memory_space>(nzOrders);
        FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> output(dim, deviceStarts, deviceDims, deviceOrders);
        return output;
    }
    
    template<>
    template<>
    FixedMultiIndexSet<Kokkos::HostSpace> FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space>::ToDevice<Kokkos::HostSpace>()
    {
        assert(false);
        return FixedMultiIndexSet<Kokkos::HostSpace>(0,0);
    }

    template<>
    template<>
    FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space>::ToDevice<Kokkos::DefaultExecutionSpace::memory_space>()
    {
        assert(false);
        return FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space>(0,0);
    }

#endif


// Explicit template instantiation
#if defined(MPART_ENABLE_GPU)
    template class mpart::FixedMultiIndexSet<Kokkos::HostSpace>;
    template class mpart::FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space>;
#else
    template class mpart::FixedMultiIndexSet<Kokkos::HostSpace>;
#endif

