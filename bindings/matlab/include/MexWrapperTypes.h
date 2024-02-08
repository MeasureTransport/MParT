#ifndef MPART_MEXWRAPPERTYPES_H
#define MPART_MEXWRAPPERTYPES_H

#include <fstream>
#include <mexplus.h>
#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include "MParT/ComposedMap.h"
#include "MParT/AffineMap.h"
#include "MParT/MapObjective.h"


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

  ConditionalMapMex(std::shared_ptr<ConditionalMapBase<MemorySpace>> init_ptr){
    map_ptr = init_ptr;
  }

  ConditionalMapMex(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks){
    map_ptr = std::make_shared<TriangularMap<MemorySpace>>(blocks);
  }

  ConditionalMapMex(unsigned int inputDim, unsigned int outputDim, unsigned int totalOrder, MapOptions opts){
    map_ptr = MapFactory::CreateTriangular<MemorySpace>(inputDim,outputDim,totalOrder,opts);
  }

  ConditionalMapMex(unsigned int inputDim, unsigned int totalOrder, StridedVector<const double, MemorySpace> centers, MapOptions opts){
    map_ptr = MapFactory::CreateSigmoidComponent<MemorySpace>(inputDim,totalOrder,centers,opts);
  }

  ConditionalMapMex(unsigned int inputDim, unsigned int outputDim, unsigned int totalOrder, StridedMatrix<const double, MemorySpace> centers, MapOptions opts){
    map_ptr = MapFactory::CreateSigmoidTriangular<MemorySpace>(inputDim,outputDim,totalOrder,centers,opts);
  }

  ConditionalMapMex(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> triMaps,std::string typeMap){
    map_ptr = std::make_shared<ComposedMap<MemorySpace>>(triMaps);
  }

  ConditionalMapMex(StridedMatrix<double,MemorySpace> A, StridedVector<double,MemorySpace> b){
    map_ptr = std::make_shared<AffineMap<MemorySpace>>(A,b);
  }

  ConditionalMapMex(StridedMatrix<double,MemorySpace> A){
    map_ptr = std::make_shared<AffineMap<MemorySpace>>(A);
  }

  ConditionalMapMex(StridedVector<double,MemorySpace> b){
    map_ptr = std::make_shared<AffineMap<MemorySpace>>(b);
  }

}; //end class

class ParameterizedFunctionMex {       // The class
public:
  std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> fun_ptr;

  ParameterizedFunctionMex(unsigned int outputDim, FixedMultiIndexSet<MemorySpace> const& mset,
                    MapOptions opts){
    fun_ptr = MapFactory::CreateExpansion<MemorySpace>(outputDim,mset,opts);
  }

  ParameterizedFunctionMex(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> init_ptr){
    fun_ptr = init_ptr;
  }
}; //end class

class MapObjectiveMex {       // The class
public:
    std::shared_ptr<MapObjective<MemorySpace>> obj_ptr;

    MapObjectiveMex(std::shared_ptr<MapObjective<MemorySpace>> init_ptr): obj_ptr(init_ptr) {};
};

// Instance manager for ConditionalMap.
template class mexplus::Session<ConditionalMapMex>;
template class mexplus::Session<ParameterizedFunctionMex>;
template class mexplus::Session<MapObjectiveMex>;


#endif // MPART_MEXWRAPPERTYPES_H