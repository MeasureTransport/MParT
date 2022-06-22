/**
TODO: ADD DESCRIPTION
 */

#include <random>

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

using namespace mpart; 

Eigen::VectorXd true_f(Eigen::VectorXd x){
    return 2*(x.array() > 2).cast<double>();
}
 
int main(int argc, char* argv[]){

    Kokkos::initialize(argc,argv);
    {

    unsigned int num_points = 1000;
    Eigen::VectorXd points;
    points.setLinSpaced(num_points, 0, 4);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 0.4);
    auto normal = [&] (int) {return distribution(generator);};

    Eigen::VectorXd noise = Eigen::VectorXd::NullaryExpr(num_points, normal);
    Eigen::VectorXd y = true_f(points) + noise;
    std::cout << y << std::endl;

//    Eigen::MatrixXi multis(2,dim);
//    multis << 0,1,
//              2,0;
//
//    MultiIndexSet mset(multis);
//
//    MapOptions opts;
//    opts.basisType    = BasisTypes::ProbabilistHermite;
//    opts.posFuncType = PosFuncTypes::SoftPlus;
//    opts.quadType    = QuadTypes::AdaptiveSimpson;
//    opts.quadAbsTol  = 1e-6;
//    opts.quadRelTol  = 1e-6;
//
//    std::shared_ptr<ConditionalMapBase> map = MapFactory::CreateComponent(mset.Fix(), opts);
//
//    unsigned int numCoeffs = mset.Size();
//    Eigen::VectorXd coeffs(numCoeffs);
//    map->SetCoeffs(coeffs); 
//
//    coeffs(0) = 1.0; // Set the linear coefficient 
//    coeffs(1) = 0.5; // Set the quadratic coefficient
//
//
//    unsigned int numPts = 128;
//    Eigen::RowMatrixXd pts = Eigen::RowMatrixXd::Random(dim,numPts);
//    Eigen::RowMatrixXd evals = map->Evaluate(pts);
//    Eigen::VectorXd logDet = map->LogDeterminant(pts);
//
//    std::cout << "Map Evaluations:\n" << evals << std::endl;
    }
    Kokkos::finalize();
	
    return 0;
}

