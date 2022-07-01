/**
TODO: ADD DESCRIPTION
 */


#include <random>
#include <fstream>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>

using namespace mpart; 

 
void LevenbergMarquadtSolver(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, 
                             Eigen::MatrixXd                                 const& x, 
                             Eigen::VectorXd                                 const& y)
{
    const unsigned int numPts = x.cols();
    assert(y.size() == numPts);

    Eigen::VectorXd coeffs = map->CoeffMap();
    Eigen::MatrixXd sens = Eigen::MatrixXd::Ones(1,numPts);
    
    Eigen::MatrixXd jac = map->CoeffGrad(x,sens);
    Eigen::VectorXd objGrad = y - map->Evaluate(x).row(0).transpose();
    double obj = 0.5*objGrad.squaredNorm();
    Eigen::VectorXd paramGrad = jac * objGrad;

    double stepSize;
    double newObj;

    const double ftol = 1e-6;
    const double gtol = 1e-4;
    double lambda = 1e-5;
    const double lambdaScale = 5;

    Eigen::VectorXd newObjGrad;
    Eigen::MatrixXd hess;

    printf("Iteration, Objective, Grad Norm,   Lambda\n");

    for(unsigned int optIt=0; optIt<5000; ++optIt){

        hess = jac * jac.transpose();
        hess += lambda * hess.diagonal().asDiagonal(); 

        map->CoeffMap() = coeffs + hess.ldlt().solve(paramGrad);
        newObjGrad = y - map->Evaluate(x).row(0).transpose();
        newObj = 0.5*newObjGrad.squaredNorm();

        if(newObj < obj){

            // Check for convergence
            if(std::abs(obj-newObj)<ftol){
                std::cout << "SUCCESS! Terminating due to small change in objective." << std::endl;
                return;
            }

            if(paramGrad.norm()<gtol){
                std::cout << "SUCCESS! Terminating due to small gradient norm." << std::endl;
                return;
            }
            
            coeffs = map->CoeffMap();
            lambda /= lambdaScale;

            objGrad = newObjGrad;// y - map->Evaluate(x).row(0).transpose();
            obj = newObj; //0.5*objGrad.squaredNorm();
            jac = map->CoeffGrad(x,sens);
            paramGrad = jac * objGrad;

        }else{
            map->CoeffMap() = coeffs;
            lambda *= lambdaScale;
        }

        printf("%9d, %9.2e, %9.2e, %6.2e\n", optIt, obj,paramGrad.norm(), lambda );
    }

}

int main(int argc, char* argv[]){

    Kokkos::initialize(argc,argv);
    {

    // Generate noisy data
    unsigned int num_points = 1000;
    int xmin = 0;
    int xmax = 4;
    Eigen::MatrixXd x(1,num_points);
    x.row(0).setLinSpaced(num_points, xmin, xmax);

    Eigen::VectorXd y_true = 2*(x.row(0).array() > 2).cast<double>();

    double noise_std = 0.4;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, noise_std);
    auto normal = [&] (int) {return distribution(generator);};
    Eigen::VectorXd y_noise = Eigen::VectorXd::NullaryExpr(num_points, normal);

    Eigen::VectorXd y_measured = y_true + y_noise;

    // Create the map
    unsigned int maxDegree = 7;
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(x.rows(), maxDegree);

    MapOptions opts;
    opts.quadMinSub = 2;
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = MapFactory::CreateComponent(mset.Fix(), opts);

    // Solve the regression problem for the map coefficients
    Eigen::VectorXd map_of_x_before = map->Evaluate(x).row(0);
    LevenbergMarquadtSolver(map, x, y_measured);
    Eigen::VectorXd map_of_x_after = map->Evaluate(x).row(0);

    // Save the data to a csv file for plotting
    std::ofstream file("data.dat");
    assert(file.is_open());

    file << "X" << "\t" << "Y_True" << "\t" << "Y_Obs" << "\t"<< "Map_Initial" << "\t" << "Map_Optimized" << "\n";
    for (size_t i = 0; i < num_points; ++i){
        file << x(i) << "\t" << y_true(i) << "\t" << y_measured(i) << "\t"<< map_of_x_before(i) << "\t" << map_of_x_after(i) << "\n";
    }

    }
    Kokkos::finalize();
	
    return 0;
}

