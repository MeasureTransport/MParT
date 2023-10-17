#ifndef MPART_ROOTFINDING_H
#define MPART_ROOTFINDING_H
#include "MParT/Utilities/Miscellaneous.h"
#include <Kokkos_Core.hpp>

namespace mpart {
namespace RootFinding {
template<typename T>
KOKKOS_INLINE_FUNCTION void swapPair(T& x1, T& x2, T& y1, T& y2) {
    simple_swap(x1, x2);
    simple_swap(y1, y2);
}

/** Finds a bracket [xlb, xub] such that f(xlb)<yd and f(xub)>yd. 
 * The info argument can be used to detect when a bracket cannot be found.  Upon exit, a value of info=0
 * indicates success while a negative value indicates failure.  info=-1 indicates that the function 
 * seems to be perfectly flat and a root might not exist.  info=-2 indicates that the maximum number of 
 * iterations (128) was exceeded.
*/
template<typename MemorySpace, typename FunctorType>
KOKKOS_INLINE_FUNCTION void FindBracket(FunctorType f,
                                        double& xlb, double& ylb,
                                        double& xub, double& yub,
                                        const double yd,
                                        int& info)
{
    const unsigned int maxIts = 128;
    double stepSize = 1.0;
    info = 0;

    ylb = f(xlb);
    yub = f(xub);

    // We actually found an upper bound...
    if(ylb>yd){
    
        mpart::simple_swap(ylb,yub);
        mpart::simple_swap(xlb,xub);

        // Now find a lower bound...
        unsigned int i;
        for(i=0; i<maxIts; ++i){ // Could just be while(true), but want to avoid infinite loop
            xlb = xub-stepSize;
            ylb = f(xlb);
            
            if(abs((yub-ylb)/(xub-xlb))<1e-12){
                info = -1;
                break;
            }

            if(ylb>yd){
                yub = ylb;
                xub = xlb;
                stepSize *= 2.0;
            }else{
                break;
            }
        }

        if(i>=maxIts)
            info = -2;

    // We have a lower bound...
    }else{
        // Now find an upper bound...
        unsigned int i;
        for(i=0; i<maxIts; ++i){ // Could just be while(true), but want to avoid infinite loop
            xub = xlb+stepSize;
            yub = f(xub);
          
            // Check to see if function is perfectly flat
            if(abs((yub-ylb)/(xub-xlb))<1e-12){
                info = -1;
                break;
            }

            if(yub<yd){
                ylb = yub;
                xlb = xub;
                stepSize *= 2.0;
            }else{
                break;
            }
        }

        if(i>=maxIts){
            info = -2;
        }
    }
 }

KOKKOS_INLINE_FUNCTION double Find_x_ITP(double xlb, double xub, double yd, double ylb, double yub,
                                         double k1, double k2, double nhalf, double n0, int it, double xtol) {

        double xb = 0.5*(xub+xlb); // bisection point
        double xf = (xub*ylb - xlb*yub)/(ylb-yub); // regula-falsi point

        double sigma = ((xb-xf)>0)?1.0:-1.0; // sign(xb-xf)
        double delta = fmin(k1*pow((xub-xlb), k2), fabs(xb-xf));

        xf += delta*sigma;

        double rho = fmin(xtol*pow(2.0, nhalf + n0 - it) - 0.5*(xub-xlb), fabs(xf - xb));
        double xc = xb - sigma*rho;
        return xc;
}

/** Computes the inverse of a function using the ITP method.  
 *  The info argument will be 0 upon successful completion and negative for failed inversions.  
 *  A value of info=-2 indicates a failure to find a bracket that contains the root. In this case, a nan will be returned.
 *  A value of info=-1 indicates that the maximum number of iterations was exceeded. 
*/
template<typename MemorySpace, typename FunctorType>
KOKKOS_INLINE_FUNCTION double InverseSingleBracket(double yd, FunctorType f, double x0, const double xtol, const double ftol, int& info)
{   
    double stepSize=1.0;
    const unsigned int maxIts = 10000;

    // First, we need to find two points that bound the solution.
    double xlb, xub;
    double ylb, yub;
    double xc, yc;
    info = 0;

    xlb = xub = x0;
    ylb = yub = f(xlb);

    // Compute initial bracket containing the root
    int bracket_info = 0;
    FindBracket<MemorySpace>(f, xlb, ylb, xub, yub, yd, bracket_info);
    if((bracket_info<0)||(((ylb>yd)||(yub<yd)))){
        info = -2;
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Bracketed search
    const double k1 = 0.1;
    const double k2 = 2.0;
    const double nhalf = ceil(log2(0.5*(xub-xlb)/xtol));
    const double n0 = 1.0;

    unsigned int it;
    for(it=0; it<maxIts; ++it){
        
        //std::cout << "Iteration " << it << std::endl;
        xc = Find_x_ITP(xlb, xub, yd, ylb, yub, k1, k2, nhalf, n0, it, xtol);

        yc = f(xc);

        if(fabs(yc-yd)<ftol){
            return xc;
        }else if(yc>yd){
            swapPair(xc, xub, yc, yub);
        }else{
            swapPair(xc, xlb, yc, ylb);
        }

        // Check for convergence
        if(((xub-xlb)<xtol)||((yub-ylb)<ftol)) break;
    };


    if(it>maxIts)
        info = -1;

    return 0.5*(xub+xlb);
}

} // RootFinding

} // mpart
#endif // MPART_ROOTFINDING_H