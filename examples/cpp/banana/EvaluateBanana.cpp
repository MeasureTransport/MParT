/**
This example shows how to construct a map from \f$\mathbb{R}^2\rightarrow \mathbb{R}\f$  with the form.
 \f[
T_2(x_1,x_2) = x_1^2 + x_2
 \f] 
 Note that this is one component of the map used to define a "banana'"
 */

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

using namespace mpart; 
 
int main(int argc, char* argv[]){

    Kokkos::initialize(argc,argv);
    {

    unsigned int dim = 2;

    Eigen::MatrixXi multis(2,dim);
    multis << 0,1,
              2,0;

    MultiIndexSet mset(multis);

    MapOptions opts;
    opts.basisType    = BasisTypes::ProbabilistHermite;
    opts.posFuncType = PosFuncTypes::SoftPlus;
    opts.quadType    = QuadTypes::AdaptiveSimpson;
    opts.quadAbsTol  = 1e-6;
    opts.quadRelTol  = 1e-6;

    std::shared_ptr<ConditionalMapBase> map = MapFactory::CreateComponent(mset.Fix(), opts);

    unsigned int numCoeffs = mset.Size();
    Eigen::VectorXd coeffs(numCoeffs);
    map->SetCoeffs(coeffs); 

    coeffs(0) = 1.0; // Set the linear coefficient 
    coeffs(1) = 0.5; // Set the quadratic coefficient


    unsigned int numPts = 128;
    Eigen::RowMatrixXd pts = Eigen::RowMatrixXd::Random(dim,numPts);
    Eigen::RowMatrixXd evals = map->Evaluate(pts);
    Eigen::VectorXd logDet = map->LogDeterminant(pts);

    std::cout << "Map Evaluations:\n" << evals << std::endl;
    }
    Kokkos::finalize();

    return 0;
}

    // // Set the coefficients of the map 
    // unsigned int numCoeffs = mset.Size();
    // Eigen::VectorXd coeffs(numCoeffs);
    // coeffs(0) = 0; // Constant term 
    // coeffs(1) = 1.0; // Linear term in second component 
    // coeffs(2) = 1.0; // Quadratic term in secon component

    // map->SetCoeffs(coeffs); // Maps to map->Coeffs() = VecToKokkos<double>(coeffs); on the c++ side.
    
    // // Updating the coeffs object will now update the coefficients used in the map
    // Eigen::RowMatrixXd pts(dim,numPts);
    // Eigen::RowMatrixXd evals = map->Evaluate(pts);
    // Eigen::VectorXd logDet = map->LogDeterminant(pts);

    // Eigen::RowMatrixXd rs = map->Inverse(evals);


//     return 0;
// }
