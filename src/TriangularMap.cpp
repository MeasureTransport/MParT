#include "MParT/TriangularMap.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
TriangularMap<MemorySpace>::TriangularMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& components, bool moveCoeffs) : ConditionalMapBase<MemorySpace>(components.back()->inputDim,
                        std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->outputDim; }),
                        std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->numCoeffs; })),
                        comps_(components)
{


    // Check the sizes of all the inputs

    for(unsigned int i=0; i<comps_.size(); ++i){
        if(comps_.at(i)->outputDim > comps_.at(i)->inputDim){
            std::stringstream msg;
            msg << "In TriangularMap constructor, the output dimension (" << comps_.at(i)->outputDim << ") of component " << i << " is greater than the input dimension (" << comps_.at(i)->inputDim << ").";
            throw std::invalid_argument(msg.str());
        }

    }

    for(unsigned int i=1; i<comps_.size(); ++i){
        if(comps_.at(i)->inputDim != (comps_.at(i-1)->inputDim + comps_.at(i)->outputDim)){
            std::stringstream msg;
            msg << "In TriangularMap constructor, the input dimension of component " << i << " is " << comps_.at(i)->inputDim;
            msg << ", but is expected to be the sum of the input dimension for component " << i-1;
            msg << "and output dimension for component " << i;
            msg << ", which is " << comps_.at(i-1)->inputDim << " + " << comps_.at(i)->outputDim << " = " << comps_.at(i-1)->inputDim + comps_.at(i)->outputDim << ".";
            throw std::invalid_argument(msg.str());
        }
    }


    // if moveCoeffs is set to true, we check if each component's coeffs are set, and then copy them into the new triangular map's coeffs
    if(moveCoeffs){

        Kokkos::View<double*,MemorySpace> coeffs("coeffs", this->numCoeffs);
        unsigned int cumNumCoeffs = 0;

        for(unsigned int i=0; i<comps_.size(); ++i){

            if(!comps_.at(i)->CheckCoefficients()){
                std::stringstream msg;
                msg << "In TriangularMap constructor, moveCoeffs set to true, but component " << i <<" doesn't have coeffs set";
                throw std::invalid_argument(msg.str());
            }

            Kokkos::View<double*,MemorySpace> subCoeffs = Kokkos::subview(coeffs, std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs));
            Kokkos::deep_copy(subCoeffs, comps_.at(i)->Coeffs());
            cumNumCoeffs += comps_.at(i)->numCoeffs;
        }

        this->WrapCoeffs(coeffs);
    }
}



template<typename MemorySpace>
void TriangularMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->WrapCoeffs( Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs)));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->WrapCoeffs( Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs)));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void TriangularMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->WrapCoeffs( Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs)));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->WrapCoeffs( Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs)));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}
#endif

template<typename MemorySpace>
void TriangularMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{
    // Evaluate the log determinant for the first component
    StridedMatrix<const double, MemorySpace> subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(0)->inputDim)), Kokkos::ALL());
    comps_.at(0)->LogDeterminantImpl(subPts, output);

    if(comps_.size()==1)
        return;

    // Vector to hold log determinant for a single component
    Kokkos::View<double*, MemorySpace> compDet("Log Determinant", output.extent(0));

    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,output.size());


    for(unsigned int i=1; i<comps_.size(); ++i){
        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        comps_.at(i)->LogDeterminantImpl(subPts, compDet);

        // Add to the output
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j){
            output(j) += compDet(j);
        });
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<double, MemorySpace> subOut;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(output, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        comps_.at(i)->EvaluateImpl(subPts, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{
    unsigned int ipdim = this->inputDim;
    unsigned int opdim = this->outputDim;
    Kokkos::View<double**, MemorySpace> fullOut("Full Output", ipdim, x1.extent(1));
    Kokkos::deep_copy(Kokkos::subview(fullOut, std::make_pair(0,int(x1.extent(0))), Kokkos::ALL()), x1);

    InverseInplace(fullOut, r);

    Kokkos::deep_copy(output, Kokkos::subview(fullOut, std::make_pair(ipdim-opdim,ipdim), Kokkos::ALL()));
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::InverseInplace(StridedMatrix<double, MemorySpace> x,
                                                StridedMatrix<const double, MemorySpace> const& r)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subR;
    StridedMatrix<const double, MemorySpace> subX;
    StridedMatrix<double, MemorySpace> subOut;

    unsigned int ipdim = ConditionalMapBase<MemorySpace>::inputDim;
    unsigned int opdim = ConditionalMapBase<MemorySpace>::outputDim;
    unsigned int extraInputs = ipdim - opdim;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        subX = Kokkos::subview(x, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subR = Kokkos::subview(r, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(x, std::make_pair(int(extraInputs + startOutDim),int(extraInputs+startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        comps_.at(i)->InverseImpl(subX, subR, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}


template<typename MemorySpace>
void TriangularMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<const double, MemorySpace> const& sens,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<const double, MemorySpace> subSens;



    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0,pts.extent(1));
    unsigned int dim = pts.extent(0);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& ptInd){
        for(unsigned int d=0; d<dim; ++d)
            output(d,ptInd) = 0.0;
    });
    Kokkos::fence();

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subSens = Kokkos::subview(sens, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        Kokkos::View<double**, MemorySpace> subOut("Component Jacobian", comps_.at(i)->inputDim, pts.extent(1));
        comps_.at(i)->GradientImpl(subPts, subSens, subOut);

        dim = comps_.at(i)->inputDim;
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& ptInd){
            for(unsigned int d=0; d<dim; ++d)
                output(d,ptInd) += subOut(d,ptInd);
        });
        Kokkos::fence();

        startOutDim += comps_.at(i)->outputDim;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<const double, MemorySpace> subSens;
    StridedMatrix<double, MemorySpace> subOut;

    int startOutDim = 0;
    int startParamDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        if(comps_.at(i)->numCoeffs != 0){

            subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
            subSens = Kokkos::subview(sens, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

            subOut = Kokkos::subview(output, std::make_pair(startParamDim,int(startParamDim+comps_.at(i)->numCoeffs)), Kokkos::ALL());
            comps_.at(i)->CoeffGradImpl(subPts, subSens, subOut);


            startParamDim += comps_.at(i)->numCoeffs;
        }

        startOutDim += comps_.at(i)->outputDim;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<double, MemorySpace> subOut;

    int startParamDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        if(comps_.at(i)->numCoeffs != 0){

            subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());

            subOut = Kokkos::subview(output, std::make_pair(startParamDim,int(startParamDim+comps_.at(i)->numCoeffs)), Kokkos::ALL());
            comps_.at(i)->LogDeterminantCoeffGradImpl(subPts, subOut);

            startParamDim += comps_.at(i)->numCoeffs;
        }
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Initialize the output to zero
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> zeroPolicy({0, 0}, {output.extent(0), output.extent(1)});
    Kokkos::parallel_for(zeroPolicy, KOKKOS_LAMBDA(const int& i, const int& j) {
        output(i,j) = 0.0;
    });

    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<double, MemorySpace> subOut;

    int numPts = pts.extent(1);
    Kokkos::View<double**,MemorySpace> compGrad("Component Gradient", this->inputDim, numPts);
    Kokkos::View<double**,MemorySpace> subGrad;

    for(unsigned int i=0; i<comps_.size(); ++i){
        int compDim = comps_.at(i)->inputDim;
        subPts = Kokkos::subview(pts, std::make_pair(0,compDim), Kokkos::ALL());
        subGrad = Kokkos::subview(compGrad, std::make_pair(0,compDim), Kokkos::ALL());

        comps_.at(i)->LogDeterminantInputGradImpl(subPts, subGrad);

        // Now accumulate the input gradient
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{compDim, numPts}});

        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& i, const int& j) {
            output(i,j) += subGrad(i,j);
        });
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> TriangularMap<MemorySpace>::Slice(int a, int b) {
    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> components;
    // TODO: Handle empty case
    if( a < 0 || a >= b || b > this->outputDim ) {
        throw std::invalid_argument("TriangularMap::Slice: 0 <= a < b <= outputDim must be satisfied.");
    }
    // Special cases if the slice is at the end of the map
    if( b <= this->comps_[0]->outputDim) {
        return this->comps_[0]->Slice(a, b);
    }
    if( a >= this->outputDim - this->comps_[this->comps_.size()-1]->outputDim) {
        unsigned int rest_of_output = this->outputDim - this->comps_[this->comps_.size()-1]->outputDim;
        return this->comps_[this->comps_.size()-1]->Slice(a - rest_of_output, b - rest_of_output);
    }

    int accum_a = 0; // Accumulated output dimension before a
    int k_a = 0; // Index of component containing a
    //TODO: Check that this is correct
    while(k_a < this->comps_.size()){
        if(accum_a + this->comps_[k_a]->outputDim > a)
            break;
        accum_a += this->comps_[k_a]->outputDim;
        k_a++;
    }

    int accum_b = accum_a; // Accumulated output dimension before b
    int k_b = k_a; // Index of component containing b
    //TODO: Check that this is correct
    while(k_b < this->comps_.size()){
        if(accum_b + this->comps_[k_b]->outputDim >= b)
            break;
        accum_b += this->comps_[k_b]->outputDim;
        k_b++;
    }

    if(k_a == k_b){
        components.push_back(this->comps_[k_a]->Slice(a-accum_a, b-accum_b));
    } else {
        components = std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>(this->comps_.begin()+k_a, this->comps_.begin()+k_b+1);
        components[0] = this->comps_[k_a]->Slice(a-accum_a, this->comps_[k_a]->outputDim);
        components[components.size()-1] = this->comps_[k_b]->Slice(0, b-accum_b);
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(components);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> TriangularMap<MemorySpace>::BlockSlice(int a, int b) {
    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> components;
    for(int k = a; k < b; k++){
        components.push_back(this->comps_[k]);
    }
    return std::make_shared<TriangularMap<MemorySpace>>(components);
}

// Explicit template instantiation
template class mpart::TriangularMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::TriangularMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif