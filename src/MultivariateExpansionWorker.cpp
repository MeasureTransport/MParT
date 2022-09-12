#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/LinearizedBasis.h"
#include "MParT/HermiteFunction.h"

using namespace mpart;

template<typename MemorySpace>
MultivariateExpansionMaxDegreeFunctor<MemorySpace>::MultivariateExpansionMaxDegreeFunctor(unsigned int dim, Kokkos::View<unsigned int*, MemorySpace> startPos, Kokkos::View<const unsigned int*, MemorySpace> maxDegreesIn) : dim(dim), startPos(startPos), maxDegrees(maxDegreesIn) 
{
};

template<typename MemorySpace>
void MultivariateExpansionMaxDegreeFunctor<MemorySpace>::operator()(const unsigned int i, unsigned int& update, const bool final) const{
    if(final)
        startPos(i) = update;

    if(i<2*dim){
        update += maxDegrees(i % dim)+1;
    }else{
        update += maxDegrees(dim-1)+1;
    }
};

template<typename MemorySpace>
CacheSizeFunctor<MemorySpace>::CacheSizeFunctor(Kokkos::View<unsigned int*, MemorySpace> startPosIn, Kokkos::View<unsigned int*, MemorySpace> cacheSizeIn) : startPos_(startPosIn), cacheSize_(cacheSizeIn){};


template<class BasisEvaluatorType, typename MemorySpace>
MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::MultivariateExpansionWorker() : dim_(0), multiSet_(FixedMultiIndexSet<MemorySpace>(1,0)){};

template<class BasisEvaluatorType, typename MemorySpace>
MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::MultivariateExpansionWorker(MultiIndexSet const& multiSet,
                                BasisEvaluatorType const& basis1d) : MultivariateExpansionWorker(multiSet.Fix(), basis1d){};

template<class BasisEvaluatorType, typename MemorySpace>
MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::MultivariateExpansionWorker(FixedMultiIndexSet<MemorySpace> const& multiSet,
                                BasisEvaluatorType const& basis1d) : dim_(multiSet.Length()),
                                                                                      multiSet_(multiSet),
                                                                                      basis1d_(basis1d),
                                                                                      startPos_("Indices for start of 1d basis evaluations", 2*multiSet.Length()+2),
                                                                                      maxDegrees_(multiSet_.MaxDegrees())
{
    Kokkos::parallel_scan(Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,2*dim_+2), MultivariateExpansionMaxDegreeFunctor<MemorySpace>(dim_,startPos_, maxDegrees_));

    // Compute the cache size and store in a double on the host
    Kokkos::View<unsigned int*, MemorySpace> dCacheSize("Temporary cache size",1);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,1), CacheSizeFunctor<MemorySpace>(startPos_, dCacheSize));
    cacheSize_ = ToHost(dCacheSize)(0);
};

template<class BasisEvaluatorType, typename MemorySpace>
void MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::FillCache1(double*          polyCache,
                                                                              StridedVector<const double,MemorySpace> pt,
                                                                              DerivativeFlags::DerivativeType derivType) const
{
    // Fill in first derivative information if needed 
    if((derivType == DerivativeFlags::Input)||(derivType==DerivativeFlags::MixedInput)){
        for(unsigned int d=0; d<dim_-1; ++d)
            basis1d_.EvaluateDerivatives(&polyCache[startPos_(d)],&polyCache[startPos_(d+dim_)], maxDegrees_(d), pt(d));

    // Evaluate all degrees of all 1d polynomials except the last dimension, which will be evaluated inside the integrand
    }else{
        for(unsigned int d=0; d<dim_-1; ++d)
            basis1d_.EvaluateAll(&polyCache[startPos_(d)], maxDegrees_(d), pt(d));
    }
}

template<class BasisEvaluatorType, typename MemorySpace>
void MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::FillCache2(double*          polyCache,
                                                                              StridedVector<const double, MemorySpace>,
                                                                              double xd,
                                                                              DerivativeFlags::DerivativeType derivType) const
{

    if((derivType==DerivativeFlags::None)||(derivType==DerivativeFlags::Parameters)){
        basis1d_.EvaluateAll(&polyCache[startPos_(dim_-1)],
                                maxDegrees_(dim_-1),
                                xd);

    }else if((derivType==DerivativeFlags::Diagonal) || (derivType==DerivativeFlags::Input)){
        basis1d_.EvaluateDerivatives(&polyCache[startPos_(dim_-1)],     // basis vals
                                        &polyCache[startPos_(2*dim_-1)],   // basis derivatives
                                        maxDegrees_(dim_-1),               // largest basis degree
                                        xd);                               // point to evaluate at

    }else if((derivType==DerivativeFlags::Diagonal2) || (derivType==DerivativeFlags::MixedInput)){
        basis1d_.EvaluateSecondDerivatives(&polyCache[startPos_(dim_-1)],     // basis vals
                                            &polyCache[startPos_(2*dim_-1)],   // basis derivatives
                                            &polyCache[startPos_(2*dim_)],     // basis second derivatives
                                            maxDegrees_(dim_-1),               // largest basis degree
                                            xd);                               // point to evaluate at
    }
}


template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::Evaluate(const double* polyCache, StridedVector<const double,MemorySpace> coeffs) const
{
    const unsigned int numTerms = multiSet_.Size();

    double output = 0.0;

    for(unsigned int termInd=0; termInd<numTerms; ++termInd)
    {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        for(unsigned int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i)
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];

        output += termVal*coeffs(termInd);
    }

    return output;
}

template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::DiagonalDerivative(const double* polyCache, StridedVector<const double,MemorySpace> coeffs, unsigned int derivOrder) const
{
    if((derivOrder==0)||(derivOrder>2)){
        assert((derivOrder==1)||(derivOrder==2));
    }

    const unsigned int numTerms = multiSet_.Size();
    double output = 0.0;

    const unsigned int posIndex = 2*dim_+derivOrder-2;

    for(unsigned int termInd=0; termInd<numTerms; ++termInd)
    {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        bool hasDeriv = false;
        for(unsigned int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i){
            if(multiSet_.nzDims(i)==dim_-1){
                termVal *= polyCache[startPos_(posIndex) + multiSet_.nzOrders(i)];
                hasDeriv = true;
            }else{
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            }

        }
        if(hasDeriv){
            // Multiply by the coefficients to get the contribution to the output
            output += termVal*coeffs(termInd);
        }
    }

    return output;
}

template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::CoeffDerivative(const double* polyCache, StridedVector<const double,MemorySpace> coeffs, StridedVector<double,MemorySpace> grad) const
{
    const unsigned int numTerms = multiSet_.Size();
    double f=0;

    for(unsigned int termInd=0; termInd<numTerms; ++termInd)
    {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        for(unsigned int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i)
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
                
        f += termVal*coeffs(termInd);
        grad(termInd) = termVal;
    }

    return f;
}


template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::InputDerivative(const double* polyCache, StridedVector<const double, MemorySpace> coeffs, StridedVector<double,MemorySpace> grad) const
{
    const unsigned int numTerms = multiSet_.Size();
    double f = 0.0;

    for(int wrt=-1; wrt<int(dim_); ++wrt){
        
        if(wrt>=0){
            grad(wrt) = 0;
        }

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i){
                if(int(multiSet_.nzDims(i))==wrt){
                    termVal *= polyCache[startPos_(dim_+wrt) + multiSet_.nzOrders(i)];
                    hasDeriv = true;
                }else{
                    termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
                }
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                grad(wrt) += termVal*coeffs(termInd);
            }else if(wrt<0){
                f += termVal*coeffs(termInd);
            }
        }
    }
    return f;
}

template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::MixedInputDerivative(const double* polyCache, StridedVector<const double, MemorySpace> coeffs, StridedVector<double, MemorySpace> grad) const
{
    const unsigned int numTerms = multiSet_.Size();
    double df = 0.0;
    unsigned int posInd;

    for(int wrt=-1; wrt<int(dim_); ++wrt){
        if(wrt>=0){
            grad(wrt) = 0;
        }

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv1 = false;
            bool hasDeriv2 = false;

            for(int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i){
                if(multiSet_.nzDims(i)==dim_-1){
                    posInd = (wrt==int(dim_-1)) ? (2*dim_) : (2*dim_-1);
                    termVal *= polyCache[startPos_(posInd) + multiSet_.nzOrders(i)];
                    
                    hasDeriv2 = true;
                    if(wrt==(dim_-1))
                        hasDeriv1 = true;

                }else if(int(multiSet_.nzDims(i))==wrt){
                    termVal *= polyCache[startPos_(dim_+wrt) + multiSet_.nzOrders(i)];
                    hasDeriv1 = true;
                }else{
                    termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
                }
            }

            // Multiply by the coefficients to get the contribution to the output
            if(hasDeriv1 && hasDeriv2){    
                grad(wrt) += termVal*coeffs(termInd);
            }else if((wrt<0) && hasDeriv2){
                df += termVal*coeffs(termInd);
            }
        }
    }
    return df;
}


template<class BasisEvaluatorType, typename MemorySpace>
double MultivariateExpansionWorker<BasisEvaluatorType, MemorySpace>::MixedCoeffDerivative(const double* cache, StridedVector<const double, MemorySpace> coeffs, unsigned int derivOrder, StridedVector<double, MemorySpace> grad) const
{
    const unsigned int numTerms = multiSet_.Size();

    if((derivOrder==0)||(derivOrder>2)){
        assert((derivOrder==1) || (derivOrder==2));
    }

    double df=0;

    const unsigned int posIndex = 2*dim_+derivOrder-2;

    // Compute coeff * polyval for each term
    for(unsigned int termInd=0; termInd<numTerms; ++termInd)
    {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        bool hasDeriv = false;
        for(unsigned int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i){
            if(multiSet_.nzDims(i)==dim_-1){
                termVal *= cache[startPos_(posIndex) + multiSet_.nzOrders(i)];
                hasDeriv = true;
            }else{
                termVal *= cache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            }

        }
        if(hasDeriv){
            // Multiply by the coefficients to get the contribution to the output
            df += termVal*coeffs(termInd);
            grad(termInd) = termVal;
        }else{
            grad(termInd) = 0.0;
        }
    }

    return df;
}


// Explicit template instantiation
template class mpart::MultivariateExpansionMaxDegreeFunctor<Kokkos::HostSpace>;
template class mpart::CacheSizeFunctor<Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<PhysicistHermite,Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<ProbabilistHermite,Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<HermiteFunction,Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<LinearizedBasis<PhysicistHermite>,Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<LinearizedBasis<ProbabilistHermite>,Kokkos::HostSpace>;
template class mpart::MultivariateExpansionWorker<LinearizedBasis<HermiteFunction>,Kokkos::HostSpace>;

#if defined(MPART_ENABLE_GPU)

    template class mpart::MultivariateExpansionMaxDegreeFunctor<mpart::DeviceSpace>;
    template class mpart::CacheSizeFunctor<mpart::DeviceSpace>;
    template class mpart::MultivariateExpansionWorker<PhysicistHermite,mpart::DeviceSpace>;
    template class mpart::MultivariateExpansionWorker<ProbabilistHermite,mpart::DeviceSpace>;
    template class mpart::MultivariateExpansionWorker<HermiteFunction,mpart::DeviceSpace>;
    template class mpart::MultivariateExpansionWorker<LinearizedBasis<PhysicistHermite>,mpart::DeviceSpace
    template class mpart::MultivariateExpansionWorker<LinearizedBasis<ProbabilistHermite>,mpart::DeviceSpace>;
    template class mpart::MultivariateExpansionWorker<LinearizedBasis<HermiteFunction>,mpart::DeviceSpace>;

#endif