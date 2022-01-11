#include "MParT/MultiIndexSet.h"

using namespace mpart;


MultiIndexSet::MultiIndexSet(unsigned int _dim, 
                             unsigned int _maxOrder) : dim(_dim)
{   
    // Figure out the number of terms in the total order
    unsigned int numTerms, numNz;
    std::tie(numTerms,numNz) = TotalOrderSize(_maxOrder, 0);
    
    // Allocate space for the multis in compressed form
    nzStarts = Kokkos::View<unsigned int*>("nzStarts", numTerms+1);
    nzDims   = Kokkos::View<unsigned int*>("nzDims", numNz);
    nzOrders = Kokkos::View<unsigned int*>("nzOrders", numNz);

    // Compute the multis
    std::vector<unsigned int> workspace(dim);
    unsigned int currNz=0;
    unsigned int currTerm=0;
    
    FillTotalOrder(_maxOrder, workspace, 0, currTerm, currNz);
}

    

std::vector<unsigned int> MultiIndexSet::GetMaxOrders() const
{   
    std::vector<unsigned int> maxOrders(dim, 0);

    for(unsigned int i=0; i<nzOrders.extent(0); ++i){
        if(nzOrders(i)>maxOrders.at(nzDims(i)))
            maxOrders.at(nzDims(i)) = nzOrders(i);
    }

    return maxOrders;
}


std::vector<unsigned int> MultiIndexSet::IndexToMulti(unsigned int index) const
{   
    std::vector<unsigned int> output(dim,0);
    
    for(unsigned int i=nzStarts(index); i<nzStarts(index+1); ++i)
        output.at( nzDims(i) ) = nzOrders(i);
    
    return output;
}

int MultiIndexSet::MultiToIndex(std::vector<unsigned int> const& multi) const
{   
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
}


void MultiIndexSet::Print() const
{
    std::cout << "Starts:\n";
    for(int i=0; i<nzStarts.extent(0); ++i)
        std::cout << nzStarts(i) << "  ";
    std::cout << std::endl;

    std::cout << "\nDims:\n";
    for(int i=0; i<nzDims.extent(0); ++i)
        std::cout << nzDims(i) << "  ";
    std::cout << std::endl;
    
    std::cout << "\nOrders:\n";
    for(int i=0; i<nzOrders.extent(0); ++i)
        std::cout << nzOrders(i) << "  ";
    std::cout << std::endl;
    

    std::cout << "\nMultis:\n";
    std::vector<unsigned int> multi;
    for(unsigned int term=0; term<nzStarts.extent(0)-1; ++term){
        multi = IndexToMulti(term);
        
        for(auto& m : multi)
            std::cout << m << "  ";
        
        std::cout << std::endl;
    }

}


unsigned int MultiIndexSet::NumTerms() const
{
    return nzStarts.extent(0)-1;
}   


std::pair<unsigned int, unsigned int> MultiIndexSet::TotalOrderSize(unsigned int maxOrder, unsigned int currDim)
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

void MultiIndexSet::FillTotalOrder(unsigned int maxOrder,
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
            for(int i=0; i<dim; ++i){
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