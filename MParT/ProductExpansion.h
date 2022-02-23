#ifndef MPART_PRODUCTEXPANSION_H
#define MPART_PRODUCTEXPANSION_H

#include "MParT/DerivativeFlags.h"

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"


namespace mpart{

/**
 @brief Defines a function in terms of the tensor product of unary basis functions.
 @details 
 - Cache memory managed elsewhere 
 \f\[
     \text{cache} = \left[\begin{array}{c}
     \phi_1^0(x_1)\\
     \phi_1^1(x_1)\\
     \vdots
     \phi_1^{p_1}\\
     \phi_2^0(x_2)\\
     \vdots\\
     \phi_2^{p_2}(x_2)\\
     \vdots \\
     \phi_d^{p_d}(x_d)\\
     \frac{\partial}{\partial x_d}\phi_d^0(x_d)\\
     \vdots\\
     \frac{\partial}{\partial x_d}\phi_d^{p_d}(x_d)\\
     \frac{\partial^2}{\partial x_d}\phi_d^0(x_d^2)\\
     \vdots\\
     \frac{\partial^2}{\partial x_d^2}\phi_d^{p_d}(x_d)
     \end{array}
     \right]
  \f\]
 
 @tparam BasisEvaluatorType The family of 1d basis functions to employ.
 */
template<class BasisEvaluatorType>
class ProductExpansion
{
public:

    ProductExpansion(MultiIndexSet const& multiSet,
                     BasisEvaluatorType const& basis1d = BasisEvaluatorType()) : ProductExpansion(multiSet.Fix(), basis1d){};

    ProductExpansion(FixedMultiIndexSet const& multiSet,    
                     BasisEvaluatorType const& basis1d = BasisEvaluatorType()) : _dim(multiSet.dim),
                                                                                 _multiSet(multiSet),
                                                                                 _basis1d(basis1d),
                                                                                 _startPos("Indices for start of 1d basis evaluations", multiSet.dim+3),
                                                                                 _maxDegrees(_multiSet.MaxDegrees())
    {   
        _startPos(0) = 0;
        for(unsigned int i=1; i<_dim+1; ++i)
            _startPos(i) = _startPos(i-1) + _maxDegrees(i-1)+1;
        _startPos(_dim+1) = _startPos(_dim) + _maxDegrees(_dim-1)+1;
        _startPos(_dim+2) = _startPos(_dim+1) + _maxDegrees(_dim-1)+1;
    }; 

    /**
     @brief Returns the size of the cache needed to evaluate the expansion (in terms of number of doubles).
     @return unsigned int  The length of the required cache vector.
     */
    unsigned int CacheSize() const{ return _startPos(_startPos.extent(0)-1);};

    /**
     @brief Returns the number of coefficients in this expansion.
     @return unsigned int The number of terms in the multiindexset, which corresponds to the number of coefficients needed to define the expansion.
     */
    unsigned int NumCoeffs() const{return _multiSet.Size();};

    /**
     @brief Precomputes parts of the cache using all but the last component of the point, i.e., using only \f$x_1,x_2,\ldots,x_{d-1}\f$, not \f$x_d\f$.
     @details 
     @tparam PointType The vector type used to define the point.  Can be anything allowing access to components with operator().  Examples are Kokkos::View<double*> or Eigen::VectorXd.  Only the first d-1 components of the vector will be accessed in this function.
     @param polyCache A pointer to the start of the cache.  This memory must be allocated before calling this function.
     @param pt The point (at least the first \f$d-1\f$ components) to use when filling in the cache.
     @param derivType

     @see FillCache2
     */
    template<typename PointType>
    void FillCache1(double*          polyCache, 
                    PointType const& pt, 
                    DerivativeFlags::DerivativeType   derivType) const
    {
        // Evaluate all degrees of all 1d polynomials except the last dimension, which will be evaluated inside the integrand
        for(unsigned int d=0; d<_dim-1; ++d)
            _basis1d.EvaluateAll(&polyCache[_startPos(d)], _maxDegrees(d), pt(d));
    }

    /**
     @brief Precomputes parts of the cache that depend on the \f$d^{th}\f$ component of the point \f$x\f$.
     @details 
     @tparam PointType The vector type used to define the point.  Can be anything allowing access to components with operator().  Examples are Kokkos::View<double*> or Eigen::VectorXd.  Only the first d-1 components of the vector will be accessed in this function.
     @param polyCache A pointer to the start of the cache.  This memory must be allocated before calling this function.
     @param pt The point to use when filling in the cache.  Should contain \f$[x_1,\ldots,x_{d-1}]\f$.  The vector itself can have more than \f$d-1\f$ components, but only the first \f$d-1\f$ will be accessed.  The value of \f$x_d\f$ is passed through the `xd` argument.
     @param xd The value of \f$x_d\f$.  This is passed separate from \f$[x_1,\ldots,x_{d-1}]\f$ to make integrating over the last component more efficient.  A copy of the point does not need to be created.
     @param derivType

     @see FillCache1
     */

    template<typename PointType>
    void FillCache2(double*          polyCache, 
                    PointType const& pt,
                    double           xd,
                    DerivativeFlags::DerivativeType derivType) const
    {   
        if(derivType==DerivativeFlags::None){
            _basis1d.EvaluateAll(&polyCache[_startPos(_dim-1)],
                                  _maxDegrees(_dim-1), 
                                  xd);

        }else if(derivType==DerivativeFlags::Diagonal){
            _basis1d.EvaluateDerivatives(&polyCache[_startPos(_dim-1)], // basis vals
                                         &polyCache[_startPos(_dim)],   // basis derivatives
                                         _maxDegrees(_dim-1),          // largest basis degree
                                         xd);                       // point to evaluate at

        }else if(derivType==DerivativeFlags::Diagonal2){
            _basis1d.EvaluateSecondDerivatives(&polyCache[_startPos(_dim-1)], // basis vals
                                               &polyCache[_startPos(_dim)],   // basis derivatives
                                               &polyCache[_startPos(_dim+1)], // basis second derivatives 
                                               _maxDegrees(_dim-1),       // largest basis degree
                                               xd);                    // point to evaluate at
        }
    }


    template<typename CoeffVecType>
    double Evaluate(const double* polyCache, CoeffVecType const& coeffs) const
    {   
        const unsigned int numTerms = _multiSet.Size();

        double output = 0.0;

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i)
                    termVal *= polyCache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
            
            output += termVal*coeffs(termInd);
        }

        return output;
    }

    /**
     * @brief Evaluates the derivative of the expansion wrt x_{D-1}
     * 
     * @tparam CoeffVecType 
     * @param polyCache 
     * @param coeffs 
     * @return double 
     */
    template<typename CoeffVecType>
    double DiagonalDerivative(const double* polyCache, CoeffVecType const& coeffs, unsigned int derivOrder) const
    {   
        if((derivOrder==0)||(derivOrder>2)){
            std::stringstream msg;
            msg << "Error in ProductExpansion::DiagonalDerivative.  The derivative order is " << derivOrder << ", but only orders of {1,2} are currently supported.";
            throw std::runtime_error(msg.str());
        }

        const unsigned int numTerms = _multiSet.Size();
        double output = 0.0;

        const unsigned int posIndex = _dim+derivOrder-1;

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                if(_multiSet.nzDims(i)==_dim-1){
                    termVal *= polyCache[_startPos(posIndex) + _multiSet.nzOrders(i)];
                    hasDeriv = true;
                }else{
                    termVal *= polyCache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                }
                
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                output += termVal*coeffs(termInd);
            }
        }

        return output;
    }

    /**
     * @brief 
     * 
     * @tparam CoeffVecType 
     * @tparam GradVecType 
     * @param polyCache 
     * @param coeffs 
     * @param grad 
     */
    template<typename CoeffVecType, typename GradVecType>
    double CoeffDerivative(const double* polyCache, CoeffVecType const& coeffs, GradVecType& grad) const
    {       
        const unsigned int numTerms = _multiSet.Size();
        double f=0;

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i)
                    termVal *= polyCache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
            
            f += termVal*coeffs(termInd);
            grad(termInd) = termVal;
        }

        return f;
    }

    template<typename CoeffVecType, typename GradVecType>
    double MixedDerivative(const double* cache, CoeffVecType const& coeffs, unsigned int derivOrder, GradVecType& grad) const
    {   
        const unsigned int numTerms = _multiSet.Size();

        if((derivOrder==0)||(derivOrder>2)){
            std::stringstream msg;
            msg << "Error in ProductExpansion::DiagonalDerivative.  The derivative order is " << derivOrder << ", but only orders of {1,2} are currently supported.";
            throw std::runtime_error(msg.str());
        }

        double df=0;
        
        const unsigned int posIndex = _dim+derivOrder-1;

        // Compute coeff * polyval for each term
        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                if(_multiSet.nzDims(i)==_dim-1){
                    termVal *= cache[_startPos(posIndex) + _multiSet.nzOrders(i)];
                    hasDeriv = true;
                }else{
                    termVal *= cache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                }
                
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                df += termVal*coeffs(termInd);
                grad(termInd) = termVal;
            }else{
                grad(termInd) = 0.0;
            }
        }

        return df;
    }
    


private:

    const unsigned int _dim;

    FixedMultiIndexSet _multiSet;
    BasisEvaluatorType _basis1d;

    Kokkos::View<unsigned int*> _startPos;
    Kokkos::View<const unsigned int*> _maxDegrees;

}; // class TensorProductExpansion

} // namespace mpart



#endif  // #ifndef MPART_TENSORPRODUCTEXPANSION_H