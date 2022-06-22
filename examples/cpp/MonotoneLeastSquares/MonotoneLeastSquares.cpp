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

double objective(Eigen::Map<Eigen::VectorXd> coeffs, std::shared_ptr<ConditionalMapBase> map, Eigen::VectorXd x, Eigen::VectorXd y, unsigned int num_points){
    map->SetCoeffs(coeffs);
    Eigen::RowMatrixXd map_of_x = map->Evaluate(x.reshaped(1,num_points));
    return (map_of_x - y.reshaped(1,num_points)).array().pow(2).sum()/num_points;
}

 
int main(int argc, char* argv[]){

    Kokkos::initialize(argc,argv);
    {

    unsigned int num_points = 1000;
    Eigen::VectorXd x;
    x.setLinSpaced(num_points, 0, 4);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 0.4);
    auto normal = [&] (int) {return distribution(generator);};

    Eigen::VectorXd noise = Eigen::VectorXd::NullaryExpr(num_points, normal);
    Eigen::VectorXd y = true_f(x) + noise;

    Eigen::MatrixXi multis(6,1);
    multis << 0,1,2,3,4,5;

    MultiIndexSet mset(multis);
    FixedMultiIndexSet fixed_mset = mset.Fix(true);

    MapOptions opts;
    std::shared_ptr<ConditionalMapBase> map = MapFactory::CreateComponent(fixed_mset, opts);
    std::cout<<map->CoeffMap()<<std::endl;

    double tmp = objective(map->CoeffMap(), map, x, y, num_points);
    std::cout<<tmp<<std::endl;

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

