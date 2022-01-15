#ifndef ORTHOGONALPOLYNOMIAL_H
#define ORTHOGONALPOLYNOMIAL_H

#include <cmath>

namespace mpart{

/*
p_{k}(x) = (a_k x + b_k) p_{k-1}(x) - c_k p_{k-2}(x)
*/
template<class Mixer>
class OrthogonalPolynomial : public Mixer
{
public:

    /* Evaluates all polynomials up to a specified order. */
    void EvaluateAll(double*              output,
                     unsigned int         maxOrder,
                     double               x) const
    {
        output[0] = this->phi0(x);
        
        if(maxOrder>0)
            output[1] = this->phi1(x);

        for(unsigned int order=2; order<=maxOrder; ++order)
            output[order] = (this->ak(order)*x + this->bk(order))*output[order-1] - this->ck(order)*output[order-2];
    }

    double Evaluate(unsigned int const order, 
                    double const x, 
                    unsigned int currDim = 0) const
    {
        if(order==0){
            return this->phi0(x);
        }else if(order==1){
            return this->phi1(x);
        }else{

            // "Downward" Clenshaw algorithm  http://mathworld.wolfram.com/ClenshawRecurrenceFormula.html
            double yk2 = 0.0;
            double yk1 = 0.0;
            double yk = 1.0;
            double alpha, beta;

            for( int k=order-1; k>=0; k-- ) {
                yk2 = yk1;
                yk1 = yk;

                alpha = this->ak(k+1)*x + this->bk(k+1);
                beta = -this->ck(k+2);
                yk = alpha*yk1 + beta*yk2;
            }
            
            beta = -this->ck(2);
            return yk1*this->phi1(x) + beta * this->phi0(x)*yk2;
        }
    }
};


class ProbabilistHermiteMixer{
public:

    double Normalization(unsigned int polyOrder) const {return sqrt(2.0*M_PI) * std::tgamma(polyOrder+1); }

protected:

    double ak(unsigned int k) const {return 1.0;}
    double bk(unsigned int k) const {return 0.0;}
    double ck(unsigned int k) const {return k-1.0;}
    double phi0(double x) const {return 1.0;}
    double phi1(double x) const {return x;}
};

typedef OrthogonalPolynomial<ProbabilistHermiteMixer> ProbabilistHermite;


class PhysicistHermiteMixer{
public:

    double Normalization(unsigned int polyOrder) const {return sqrt(M_PI) * pow(2.0, static_cast<double>(polyOrder)) * std::tgamma(polyOrder+1); }

protected:

    double ak(unsigned int k) const {return 2.0;}
    double bk(unsigned int k) const {return 0.0;}
    double ck(unsigned int k) const {return 2.0*(k-1.0);}
    double phi0(double x) const {return 1.0;}
    double phi1(double x) const {return 2.0*x;}
};

typedef OrthogonalPolynomial<PhysicistHermiteMixer> PhysicistHermite;


} // namespace mpart

#endif