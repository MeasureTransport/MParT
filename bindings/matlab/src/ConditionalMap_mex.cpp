#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "MexMapOptionsConversions.h"
#include "MParT/MapOptions.h"
#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include <Eigen/Dense>


#include <chrono>


using namespace mpart;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;

class ConditionalMapMex {       // The class
public:             
  std::shared_ptr<ConditionalMapBase<MemorySpace>> map_ptr;

  ConditionalMapMex(FixedMultiIndexSet<MemorySpace> const& mset, 
                    MapOptions                             opts){
    map_ptr = MapFactory::CreateComponent<MemorySpace>(mset,opts);
  }

  ConditionalMapMex(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks){
    map_ptr = std::make_shared<TriangularMap<MemorySpace>>(blocks);
  }
  ConditionalMapMex(unsigned int inputDim, unsigned int outputDim, unsigned int totalOrder, MapOptions opts){
    map_ptr = MapFactory::CreateTriangular<MemorySpace>(inputDim,outputDim,totalOrder,opts);
  }
}; //end class

// Instance manager for ConditionalMap.
template class mexplus::Session<ConditionalMapMex>;

namespace {


MEX_DEFINE(ConditionalMap_newTriMap) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
  
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);

  std::vector<intptr_t> list_id = input.get<std::vector<intptr_t>>(0);
  unsigned int numBlocks = list_id.size();
  
  std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> blocks(numBlocks);
  for(unsigned int i=0;i<numBlocks;++i){
      const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(list_id.at(i)); 
      blocks.at(i) = condMap.map_ptr;
    }
  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(blocks)));
}

MEX_DEFINE(ConditionalMap_newTotalTriMap) (int nlhs, mxArray* plhs[],
                                           int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 12);
  OutputArguments output(nlhs, plhs, 1);
  unsigned int inputDim = input.get<unsigned int>(0);
  unsigned int outputDim = input.get<unsigned int>(1);
  unsigned int totalOrder = input.get<unsigned int>(2);

  MapOptions opts = MapOptionsFromMatlab(input.get<std::string>(3),input.get<std::string>(4),
                                         input.get<std::string>(5),input.get<double>(6),
                                         input.get<double>(7),input.get<unsigned int>(8),
                                         input.get<unsigned int>(9),input.get<unsigned int>(10),
                                         input.get<bool>(11));

  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(inputDim,outputDim,totalOrder,opts)));
}

MEX_DEFINE(ConditionalMap_newMap) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 10);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  MapOptions opts = MapOptionsFromMatlab(input.get<std::string>(1),input.get<std::string>(2),
                                         input.get<std::string>(3),input.get<double>(4),
                                         input.get<double>(5),input.get<unsigned int>(6),
                                         input.get<unsigned int>(7),input.get<unsigned int>(8),
                                         input.get<bool>(9));

  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(mset.Fix(),opts)));
}

// Defines MEX API for delete.
MEX_DEFINE(ConditionalMap_deleteMap) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<ConditionalMapMex>::destroy(input.get(0));
}

MEX_DEFINE(ConditionalMap_SetCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto coeffs = MexToKokkos1d(prhs[1]);
  condMap.map_ptr->SetCoeffs(coeffs);
}

MEX_DEFINE(ConditionalMap_GetComponent) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  TriangularMap<MemorySpace> *triMap = Session<TriangularMap<MemorySpace>>::get(input.get(0));
  unsigned int i = input.get<unsigned int>(1);
  output.set(0, Session<std::shared_ptr<ConditionalMapBase<MemorySpace>>>::create(new std::shared_ptr<ConditionalMapBase<MemorySpace>>(triMap->GetComponent(i))));
}

MEX_DEFINE(ConditionalMap_Coeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto coeffs = KokkosToVec(condMap.map_ptr->Coeffs());
  output.set(0,coeffs);
}

MEX_DEFINE(ConditionalMap_numCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto numcoeffs = condMap.map_ptr->numCoeffs;
  output.set(0,numcoeffs);
}

MEX_DEFINE(ConditionalMap_outputDim) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  unsigned int outDim = condMap.map_ptr->outputDim;
  output.set(0, outDim);
}

MEX_DEFINE(ConditionalMap_inputDim) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  unsigned int inDim = condMap.map_ptr->inputDim;
  output.set(0, inDim);
}

MEX_DEFINE(ConditionalMap_Evaluate) (int nlhs, mxArray* plhs[],
                                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 0);

  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  StridedMatrix<const double, Kokkos::HostSpace> pts = MexToKokkos2d(prhs[1]);
  StridedMatrix<double, Kokkos::HostSpace> out = MexToKokkos2d(prhs[2]); 
  condMap.map_ptr->EvaluateImpl(pts, out);
}

MEX_DEFINE(ConditionalMap_LogDeterminant) (int nlhs, mxArray* plhs[],
                                           int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 0);

  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  
  Kokkos::View<double*, Kokkos::HostSpace> out = MexToKokkos1d(prhs[2]);  
  StridedMatrix<const double, Kokkos::HostSpace> pts = MexToKokkos2d(prhs[1]);

  condMap.map_ptr->LogDeterminantImpl(pts, out);
 }

MEX_DEFINE(ConditionalMap_Inverse) (int nlhs, mxArray* plhs[],
                                    int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 4);
  OutputArguments output(nlhs, plhs, 0);

  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  
  auto x1 = MexToKokkos2d(prhs[1]);
  auto r = MexToKokkos2d(prhs[2]);
  auto inv = MexToKokkos2d(prhs[3]);

  condMap.map_ptr->InverseImpl(x1,r, inv);
}

MEX_DEFINE(ConditionalMap_CoeffGrad) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 4);
  OutputArguments output(nlhs, plhs, 0);

  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));

  auto pts = MexToKokkos2d(prhs[1]);
  auto sens = MexToKokkos2d(prhs[2]);
  auto out = MexToKokkos2d(prhs[3]);
  
  condMap.map_ptr->CoeffGradImpl(pts,sens,out);
}

MEX_DEFINE(ConditionalMap_LogDeterminantCoeffGrad) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 0);

  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));

  auto pts = MexToKokkos2d(prhs[1]);
  auto out = MexToKokkos2d(prhs[2]);
  
  condMap.map_ptr->LogDeterminantCoeffGradImpl(pts,out);
}

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.