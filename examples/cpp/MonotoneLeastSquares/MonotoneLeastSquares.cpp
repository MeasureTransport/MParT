/**
TODO: ADD DESCRIPTION
 */

#include <random>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

using namespace mpart; 

struct Args{
    std::shared_ptr<ConditionalMapBase> map;
    Eigen::VectorXd x;
    Eigen::VectorXd y;
    unsigned int num_points;
};

Eigen::VectorXd true_f(Eigen::VectorXd x){
    return 2*(x.array() > 2).cast<double>();
}

inline double objective(Eigen::VectorXd coeffs, Eigen::VectorXd* grad_out, void* opt_data){
    Args* args = (Args*)opt_data;
    std::shared_ptr<ConditionalMapBase> map = args->map;
    Eigen::VectorXd x = args->x;
    Eigen::VectorXd y = args->y;
    unsigned int num_points = args->num_points;

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

    Args* args = new Args;
    args->map = map;
    args->x = x;
    args->y = y;
    args->num_points = num_points;

    double error_before = objective(map->CoeffMap(), nullptr, args);
    std::cout<<"Initial error \t= "<<error_before<<std::endl;

    Eigen::VectorXd initial = map->CoeffMap();
    
    bool success = optim::nm(initial, objective, args);
    std::cout<<"Optimization successfull? "<<success<<std::endl;

    double error_after = objective(map->CoeffMap(), nullptr, args);
    std::cout<<"Final error \t= "<<error_after<<std::endl;
    }
    Kokkos::finalize();
	
    return 0;
}

